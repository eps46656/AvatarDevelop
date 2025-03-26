import argparse
import dataclasses
import os
import pathlib
import shlex
import sqlite3
import time
import traceback
import typing

import timedinput
import torch
from beartype import beartype

from . import dataset_utils, sqlite_utils, utils


@beartype
@dataclasses.dataclass
class CheckpointMeta:
    id: int
    prev: int
    epochs_cnt: int
    message: typing.Optional[str]
    deep_saved: bool
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

        self.conn.execute("PRAGMA foreign_keys = ON")
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
                    serialize=lambda x: None if x == -1 else x,
                    deserialize=lambda x: -1 if x is None else x,),
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

    def Eval(self):
        raise utils.UnimplementationError()


@beartype
class AutoSavingConfig:
    min_diff_time: int
    max_diff_time: int
    min_epochs_cnt: int
    max_epochs_cnt: int

    def Check(self):
        assert 0 <= self.min_diff_time
        assert self.min_diff_time <= self.max_diff_time

        assert 0 <= self.min_epochs_cnt
        assert self.min_epochs_cnt <= self.max_epochs_cnt

    def QuerySave(self, diff_time: int, diff_epochs_cnt: int):
        if self.max_diff_time <= diff_time or self.max_epochs_cnt <= diff_epochs_cnt:
            return True

        if diff_time < self.min_diff_time or diff_epochs_cnt < self.min_epochs_cnt:
            return False

        return True


"""

diff_time = cur_time - prv_time
diff_epochs_cnt = cur_epochs_cnt - prv_epochs_cnt

if max_time <= diff_time or max_epochs_cnt <= diff_epochs_cnt:
    return save

if diff_time < min_time or diff_epochs_cnt < min_epochs_cnt
    return not save

return save

"""


@beartype
def MakeCommandParser():
    parser = argparse.ArgumentParser(prog="parser")

    subparser = parser.add_subparsers(
        dest="op",
        required=True,
    )

    # ---

    nop_parser = subparser.add_parser(
        "nop",
        help="nop",
    )

    # ---

    show_parser = subparser.add_parser(
        "show",
        help="show status",
    )

    # ---

    load_parser = subparser.add_parser(
        "load",
        help="load a checkpoint",
    )

    load_parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="the target checkpoint's id",
    )

    # ---

    load_latest_parser = subparser.add_parser(
        "load_latest",
        help="load the latest checkpoint",
    )

    # ---

    save_parser = subparser.add_parser(
        "save",
        help="save current state to a checkpoint",
    )

    save_parser.add_argument(
        "--deep_saved",
        action="store_true",
        help="save the total state",
    )

    # ---

    train_parser = subparser.add_parser(
        "train",
        help="train the module",
    )

    train_parser.add_argument(
        "--epochs_cnt",
        type=int,
        default=1,
        help="the number of epochs to train",
    )

    train_parser.add_argument(
        "--min_diff_time",
        type=int,
        default=10 * 60,
        help="",
    )

    train_parser.add_argument(
        "--max_diff_time",
        type=int,
        default=30 * 60 * 60,
        help="",
    )

    train_parser.add_argument(
        "--min_diff_epochs_cnt",
        type=int,
        default=20,
        help="",
    )

    train_parser.add_argument(
        "--max_diff_epochs_cnt",
        type=int,
        default=100,
        help="",
    )

    train_parser.add_argument(
        "--loop",
        action="store_true",
        help="",
    )

    # ---

    eval_parser = subparser.add_parser(
        "eval",
        help="eval the module",
    )

    # ---

    exe_cmd_file_parser = subparser.add_parser(
        "exe_cmd_file",
        help="execute file commands",
    )

    # ---

    exit_parser = subparser.add_parser(
        "exit",
        help="exit the cli mode",
    )

    return parser


@beartype
class Trainer:
    @staticmethod
    def _GetLogDatabasePath(proj_dir: pathlib.Path):
        return proj_dir / "log.db"

    @staticmethod
    def _GetCheckpointDataPath(proj_dir: pathlib.Path, id: int):
        return proj_dir / f"ckpt_data_{id}.pth"

    @staticmethod
    def _GetCancelTokenPath(proj_dir: pathlib.Path, id: int):
        return proj_dir / f"cancel_token.txt"

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

        self.__prv: int = -1
        self.__epochs_cnt: int = 0
        self.__avg_loss: float = 0.0

        self.__diff_epochs_cnt = 0

        self.training_core: TrainingCore = None

        utils.WriteFile(Trainer._GetCancelTokenPath(self.__proj_dir), "w", "")

    def GetTrainingCore(self):
        return self.training_core

    def SetTrainingCore(self, training_core: TrainingCore):
        self.training_core = training_core

    def Show(self):
        print(f"")
        print(f"            prv = {self.__prv}")
        print(f"     epochs cnt = {self.__epochs_cnt}")
        print(f"diff epochs cnt = {self.__diff_epochs_cnt}")
        print(f"       avg loss = {self.__avg_loss}")
        print(f"")

    def GetatestCheckpointMeta(self):
        return self.__log_db.SelectLatestCheckpointMeta()

    def Load(self, id: int):
        assert self.training_core is not None

        ckpt_meta = self.__log_db.SelectCheckpointMeta({"id": id})

        ckpt_data_path = Trainer._GetCheckpointDataPath(
            self.__proj_dir, ckpt_meta.id)

        d = torch.load(ckpt_data_path)

        if ckpt_meta.deep_saved:
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
        deep_saved: bool = False,
    ):
        assert self.training_core is not None

        id = int(time.time())

        self.__log_db.InsertCheckpointMeta(CheckpointMeta(
            id=id,
            prev=self.__prv,
            epochs_cnt=self.__epochs_cnt,
            message=message,
            deep_saved=deep_saved,
            avg_loss=self.__avg_loss,
        ))

        self.__prv = id
        self.__diff_epochs_cnt = 0

        d = dict()

        if deep_saved:
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

        self.__epochs_cnt += 1
        self.__diff_epochs_cnt += 1
        self.__avg_loss = training_result.avg_loss

    def Train(
        self,
        epochs_cnt: int,
        min_diff_time: int,
        max_diff_time: int,
        min_diff_epochs_cnt: int,
        max_diff_epochs_cnt: int,
    ):
        assert 0 <= epochs_cnt

        assert 0 <= min_diff_time
        assert min_diff_time <= max_diff_time

        assert 0 <= min_diff_epochs_cnt
        assert min_diff_epochs_cnt <= max_diff_epochs_cnt

        cancel_token_path = Trainer._GetCancelTokenPath(self.__proj_dir)

        utils.WriteFile(cancel_token_path, "w", "")

        for _ in range(epochs_cnt):
            self._Train()

            cur_time = int(time.time())

            diff_time = cur_time - self.__prv

            if max_diff_time <= diff_time or max_diff_epochs_cnt <= self.__diff_epochs_cnt:
                self.Save()
                continue

            if diff_time < max_diff_time or self.__diff_epochs_cnt < max_diff_epochs_cnt:
                continue

            self.Save()

            if len(utils.ReadFile(cancel_token_path, "r")) != 0:
                break

    def Eval(self):
        self.training_core.Eval()

    def _CmdFileHandler(self, parser):
        pass

    def _CLIHandler(self, parser):
        cmd = shlex.split(input("trainer> "))

        args = parser.parse_args(cmd)

        match args.op:
            case "show":
                self.Show()

            case "load":
                self.Load(args.id)

            case "load_latest":
                self.LoadLatest()

            case "save":
                self.Save(args.deep_saved)

            case "train":
                self.Train(
                    epochs_cnt=args.epochs_cnt,
                    min_diff_time=args.min_diff_time,
                    max_diff_time=args.max_diff_time,
                    min_diff_epochs_cnt=args.min_diff_epochs_cnt,
                    max_diff_epochs_cnt=args.max_diff_epochs_cnt,
                )

            case "eval":
                self.Eval()

            case "exit":
                return False

        return True

    def EnterCLI(self):
        parser = MakeCommandParser()

        while True:
            try:
                if not self._CLIHandler(parser):
                    return
            except:
                print(traceback.format_exc())
                continue
