import dataclasses
import sqlite3
import traceback
import typing

from beartype import beartype


@beartype
def make_cmd_cond_param(conditions: dict[str, object]):
    return " AND ".join(f"{col_name} = ?" for col_name in conditions.keys()), tuple(conditions.values())


@beartype
def count(
    cursor: sqlite3.Cursor,
    table_name: str,
    conditions: typing.Optional[dict[str, object]],
):
    cmd = [
        f"SELECT COUNT(*)",
        f"FROM {table_name}",
    ]

    cmd_params = tuple()

    if conditions:
        cmd_conds, cmd_params = make_cmd_cond_param(conditions)
        cmd.append(f"WHERE {cmd_conds}")

    cursor.execute(" ".join(cmd), cmd_params)

    return cursor.fetchone()[0]


@beartype
def select(
    cursor: sqlite3.Cursor,
    table_name: str,
    fetching_cols: typing.Iterable[str],
    conditions: typing.Optional[dict[str, object]],
):
    cmd_fetching_cols = ", ".join(fetching_cols)

    cmd = [
        f"SELECT {cmd_fetching_cols}",
        f"FROM {table_name}",
    ]

    cmd_params = tuple()

    if conditions:
        cmd_conds, cmd_params = make_cmd_cond_param(conditions)
        cmd.append(f"WHERE {cmd_conds}")

    cursor.execute(" ".join(cmd), cmd_params)


@beartype
def insert(
    cursor: sqlite3.Cursor,
    table_name: str,
    row: dict[str, object],
):
    assert row

    col_names = ", ".join(row.keys())
    col_vals = tuple(row.values())
    col_placeholders = ", ".join("?" for _ in row.keys())

    cursor.execute(f"""
        INSERT OR REPLACE
        INTO {table_name}
        ({col_names}) VALUES ({col_placeholders})
    """, col_vals)

    return cursor.rowcount == 1


@beartype
def delete(
    cursor: sqlite3.Cursor,
    table_name: str,
    fetching_cols: typing.Optional[typing.Iterable[str]],
    conditions: typing.Optional[dict[str, object]],
):
    assert conditions

    cmd = [
        f"DELETE",
        f"FROM {table_name}",
    ]

    cmd_params = tuple()

    if conditions is not None:
        cmd_conds, cmd_params = make_cmd_cond_param(conditions)
        cmd.append(f"WHERE {cmd_conds}")

    if fetching_cols is not None:
        cmd_fetching_cols = ", ".join(fetching_cols) if fetching_cols else None
        cmd.append(f"RETURNING {cmd_fetching_cols}")

    cursor.execute(" ".join(cmd), cmd_params)


@dataclasses.dataclass
class DataTableColAttr:
    serialize: typing.Callable[[object], object]
    deserialize: typing.Callable[[object], object]


@beartype
class DataTable:
    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        col_attrs: dict[str, DataTableColAttr],
    ):
        self.conn = conn
        self.table_name = table_name

        self.col_attrs: dict[str, DataTableColAttr] = col_attrs

    def _serialize_conds(
        self,
        conditions: typing.Optional[dict[str, object]],
    ):
        if conditions is None:
            return None

        for col_name in conditions.keys():
            assert col_name in self.col_attrs

        return {
            col_name: self.col_attrs[col_name].serialize(col_val)
            for col_name, col_val in conditions.items()
        }

    def _serialize_row(self, row: dict[str, object]):
        ret = dict()

        for col_name, col_attr in self.col_attrs.items():
            try:
                col_val = row[col_name]
            except KeyError:
                continue

            ret[col_name] = col_attr.serialize(col_val)

        return ret

    def row_to_dict(self, row):
        return {
            col_name: col_attr.deserialize(col_val)
            for col_name, col_attr, col_val in zip(self.col_attrs.keys(), self.col_attrs.values(), row)
        }

    def count(self, conditions: typing.Optional[dict[str, object]]):
        try:
            cursor = self.conn.cursor()

            return count(
                cursor,
                self.table_name,
                conditions,
            )
        except:
            print(traceback.format_exc())
            self.conn.rollback()

    def _select(self, conditions: typing.Optional[dict[str, object]]):
        conditions = self._serialize_conds(conditions)

        cursor = self.conn.cursor()

        select(
            cursor,
            self.table_name,
            self.col_attrs.keys(),
            conditions,
        )

        return cursor

    def select_many(self, conditions: typing.Optional[dict[str, object]]):
        cursor = self._select(conditions)

        return [self.row_to_dict(row) for row in cursor.fetchall()]

    def select_one(self, conditions: typing.Optional[dict[str, object]]):
        cursor = self._select(conditions)

        row = cursor.fetchone()

        return None if row is None else self.row_to_dict(row)

    def insert(self, row: dict[str, object]):
        row = self._serialize_row(row)

        try:
            cursor = self.conn.cursor()

            ret = insert(cursor, self.table_name, row)

            self.conn.commit()

            return ret
        except:
            print(traceback.format_exc())
            self.conn.rollback()

    def delete(
        self,
        conditions: typing.Optional[dict[str, object]],
        fetching: bool,
    ):
        conditions = self._serialize_conds(conditions)

        try:
            cursor = self.conn.cursor()

            if not fetching:
                delete(cursor, self.table_name, None, conditions)
                self.conn.commit()
                return None

            delete(
                cursor,
                self.table_name,
                self.col_attrs.keys(),
                conditions,
            )

            ret = [self.row_to_dict(row) for row in cursor.fetchall()]

            self.conn.commit()

            return ret
        except sqlite3.Error:
            print(traceback.format_exc())
            self.conn.rollback()

    def delete_all(self):
        try:
            cursor = self.conn.cursor()

            delete(cursor, self.table_name, None, None)

            self.conn.commit()
        except sqlite3.Error:
            print(traceback.format_exc())
            self.conn.rollback()
