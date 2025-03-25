import dataclasses
import os
import pathlib
import sqlite3
import time
import traceback
import typing

import torch
import tqdm
from beartype import beartype

from . import dataset_utils, sqlite_utils, utils


@beartype
@dataclasses.dataclass
class CheckpointMeta:
    id: int
    prev: int
    epochs_cnt: int
    message: typing.Optional[str]
    deep_save: bool
    avg_loss: float


@beartype
class LogDatabase:
    def __init__(self, db_path: os.PathLike):
        super().__init__()

        db_path = pathlib.Path(db_path)

        is_new_db = not db_path.exists()

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)

        if is_new_db:
            cursor = self.conn.cursor()

            cursor.execute("""
                CREATE TABLE CheckpointMeta(
                    id UNSIGNED BIGINT PRIMARY KEY NOT NULL,

                    prev UNSIGNED BIGINT NULL,

                    deep_saved INT NOT NULL,

                    epochs_cnt UNSIGNED INT,

                    message TEXT,

                    avg_loss REAL,

                    FOREIGN KEY (prev) REFERENCES CheckpointMeta(id)
                        ON DELETE SET NULL
                );
            """)

        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -524288")

        self.ckpt_meta_table = sqlite_utils.SqliteDataTable(
            self.conn,
            "CheckpointMeta",
            {
                "id": sqlite_utils.SqliteDataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "prev": sqlite_utils.SqliteDataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "epochs_cnt": sqlite_utils.SqliteDataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "message": sqlite_utils.SqliteDataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "deep_saved": sqlite_utils.SqliteDataTableColAttr(
                    serialize=lambda x: 1 if x else 0,
                    deserialize=lambda x: x != 0,),
                "avg_loss": sqlite_utils.SqliteDataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
            }
        )

    def SelectCheckpointMeta(
        self,
        conditions: typing.Optional[dict[str, object]],
    ):
        account = self.ckpt_meta_table.SelectOne(conditions)
        return None if account is None else CheckpointMeta(**account)

    def SelectLatestCheckpointMeta(self):
        table_name = "CheckpointMeta"

        cmd = f"SELECT * FROM {table_name} WHERE id = (SELECT MAX(id) FROM {table_name})"

        try:
            cursor = self.ckpt_meta_table.conn.cursor()

            cursor.execute(cmd)

            row = cursor.fetchone()

            return None if row is None else \
                CheckpointMeta(**self.ckpt_meta_table.RowToDict(row))
        except:
            print(traceback.format_exc())
            self.ckpt_meta_table.conn.rollback()

    def InsertCheckpointMeta(self, ckpt_meta: CheckpointMeta):
        return self.ckpt_meta_table.Insert(ckpt_meta.__dict__)

    def DeleteCheckpointMetas(
        self,
        conditions: typing.Optional[dict[str, object]],
        fetching: bool,
    ):
        ckpt_metas = self.ckpt_meta_table.Delete(conditions, fetching)
        return None if ckpt_metas is None else [
            CheckpointMeta(**ckpt_meta) for ckpt_meta in ckpt_metas]


@beartype
@dataclasses.dataclass
class TrainingResult:
    avg_loss: float


@beartype
@dataclasses.dataclass
class TrainingCore:
    module: typing.Optional[torch.nn.Module]
    dataset: dataset_utils.Dataset
    dataset_loader: dataset_utils.DatasetLoader
    loss_func: typing.Callable
    optimizer: typing.Optional[torch.optim.Optimizer]
    scheduler: typing.Optional[object]

    def Train(self) -> TrainingResult:
        raise utils.UnimplementationError()


@beartype
class Trainer:
    @staticmethod
    def _GetLogDatabasePath(proj_dir: pathlib.Path):
        return proj_dir / "log.db"

    @staticmethod
    def _GetCheckpointDataPath(proj_dir: pathlib.Path, id: int):
        return proj_dir / f"ckpt_data_{id}.pth"

    def __init__(
        self,
        proj_dir: os.PathLike,
    ):
        self.__proj_dir = pathlib.Path(proj_dir)

        if not self.__proj_dir.exists():
            os.makedirs(self.__proj_dir, exist_ok=True)
        else:
            assert self.__proj_dir.is_dir()

        log_db_path = Trainer._GetLogDatabasePath(self.__proj_dir)

        self.__log_db = LogDatabase(log_db_path)

        self.__cur_previous = -1
        self.__cur_epochs_cnt = 0
        self.__cur_avg_loss = 0.0

        self.__diff_epochs_cnt = 0

        self.training_core: TrainingCore = None

    # ---

    def __del__(self):
        pass

    def GetTrainingCore(self):
        return self.training_core

    def SetTrainingCore(self, training_core: TrainingCore):
        self.training_core = training_core

    def GetatestCheckpointMeta(self):
        return self.__log_db.SelectLatestCheckpointMeta()

    def Load(self, id: int):
        assert self.training_core is not None

        ckpt_meta = self.__log_db.SelectCheckpointMeta({"id": id})

        ckpt_data_path = Trainer._GetCheckpointDataPath(
            self.__proj_dir, ckpt_meta.id)

        d = torch.load(ckpt_data_path)

        if ckpt_meta.deep_save:
            for field_name in {"module", "optimizer", "scheduler"}:
                if field_name in d:
                    setattr(self.training_core, field_name, d[field_name])
        else:
            for field_name in {"module", "optimizer", "scheduler"}:
                if field_name not in d:
                    continue

                field_val = getattr(self.training_core, field_name)
                assert field_val is not None

                field_val.load_state_dict(d[field_name])

    def LoadLatest(self):
        ckpt_meta = self.GetatestCheckpointMeta()
        self.Load(ckpt_meta.id)

    def Save(
        self,
        message: str = None,
        deep_save: bool = False,
    ):
        assert self.training_core is not None

        id = int(time.time())

        self.__log_db.InsertCheckpointMeta(CheckpointMeta(
            id=id,
            prev=self.__cur_previous,
            epochs_cnt=self.__cur_epochs_cnt,
            message=message,
            deep_save=deep_save,
            avg_loss=self.__cur_avg_loss,
        ))

        self.__cur_previous = id
        self.__diff_epochs_cnt = 0

        d = dict()

        if deep_save:
            for field_name in {"module", "optimizer", "scheduler"}:
                field_val = getattr(self.training_core, field_name)

                d[field_name] = None if field_val is None else \
                    field_val.state_dict()
        else:
            d = {
                field_name: getattr(self.training_core, field_name)
                for field_name in {"module", "optimizer", "scheduler"}
            }

        torch.save(
            d, Trainer._GetCheckpointDataPath(self.__proj_dir, id))

    def _Train(self):
        training_result = self.training_core.Train()

        self.__cur_epochs_cnt += 1
        self.__diff_epochs_cnt += 1
        self.__cur_avg_loss = training_result.avg_loss

    def Train(self, epochs_cnt: int):
        assert 0 <= epochs_cnt

        if epochs_cnt == 0:
            return

        for _ in range(epochs_cnt):
            self._Train()
