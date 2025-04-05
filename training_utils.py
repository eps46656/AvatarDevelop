import argparse
import dataclasses
import os
import pathlib
import shlex
import sqlite3
import sys
import time
import traceback
import typing

import torch
from beartype import beartype

from . import sqlite_utils, utils


@beartype
@dataclasses.dataclass
class CheckpointMeta:
    id: int
    prv: int
    epochs_cnt: int
    message: typing.Optional[str]
    full: bool
    avg_loss: float


@beartype
class LogDatabase:
    def __init__(self, db_path: os.PathLike):
        super().__init__()

        db_path = utils.to_pathlib_path(db_path)

        is_new_db = not db_path.exists()

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)

        if is_new_db:
            cursor = self.conn.cursor()

            cursor.execute("""
                CREATE TABLE CheckpointMeta(
                    id UNSIGNED BIGINT PRIMARY KEY NOT NULL,

                    prv UNSIGNED BIGINT NULL,

                    full INT NOT NULL,

                    epochs_cnt UNSIGNED INT,

                    message TEXT,

                    avg_loss REAL,

                    FOREIGN KEY (prv) REFERENCES CheckpointMeta(id)
                        ON DELETE SET NULL
                );
            """)

        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -524288")

        self.ckpt_meta_table = sqlite_utils.DataTable(
            self.conn,
            "CheckpointMeta",
            {
                "id": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "prv": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: None if x == -1 else x,
                    deserialize=lambda x: -1 if x is None else x,),
                "epochs_cnt": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "message": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "full": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: 1 if x else 0,
                    deserialize=lambda x: x != 0,),
                "avg_loss": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
            }
        )

    def select_ckpt_meta(
        self,
        conditions: typing.Optional[dict[str, object]],
    ):
        ckpt_meta = self.ckpt_meta_table.select_one(conditions)
        return None if ckpt_meta is None else CheckpointMeta(**ckpt_meta)

    def select_latest_ckpt_meta(self):
        table_name = "CheckpointMeta"

        cmd = f"SELECT MAX(id) FROM {table_name}"

        try:
            cursor = self.ckpt_meta_table.conn.cursor()

            cursor.execute(cmd)

            row = cursor.fetchone()

            return None if row is None else self.select_ckpt_meta({"id": row[0]})
        except:
            print(traceback.format_exc())
            self.ckpt_meta_table.conn.rollback()

    def insert_ckpt_meta(self, ckpt_meta: CheckpointMeta):
        return self.ckpt_meta_table.insert(ckpt_meta.__dict__)

    def delete_ckpt_metas(
        self,
        conditions: typing.Optional[dict[str, object]],
        fetching: bool,
    ):
        ckpt_metas = self.ckpt_meta_table.delete(conditions, fetching)
        return None if ckpt_metas is None else [
            CheckpointMeta(**ckpt_meta) for ckpt_meta in ckpt_metas]


@beartype
@dataclasses.dataclass
class TrainingResult:
    avg_loss: float


@beartype
class TrainingCore:
    @property
    def module(self) -> typing.Optional[torch.nn.Module]:
        raise utils.UnimplementationError()

    @module.setter
    def module(self, module: typing.Optional[torch.nn.Module]):
        raise utils.UnimplementationError()

    # ---

    @property
    def optimizer(self) -> typing.Optional[torch.optim.Optimizer]:
        raise utils.UnimplementationError()

    @optimizer.setter
    def optimizer(self, optimizer: typing.Optional[torch.optim.Optimizer]):
        raise utils.UnimplementationError()

    # ---

    @property
    def scheduler(self) -> object:
        raise utils.UnimplementationError()

    @optimizer.setter
    def scheduler(self, scheduler: object):
        raise utils.UnimplementationError()

    # ---

    def get(self, field_name: str):
        assert field_name in {"module", "optimizer", "scheduler"}

        ret = getattr(self, field_name)

        print(f"{ret=}")

        return ret

    def set(self, field_name: str, field_val: object):
        assert field_name in {"module", "optimizer", "scheduler"}
        return setattr(self, field_name, field_val)

    # ---

    def train(self) -> TrainingResult:
        raise utils.UnimplementationError()

    def eval(self):
        raise utils.UnimplementationError()


DEFAULT_DIFF_TIME_WEIGHT = 1 / (30 * 60)
DEFAULT_DIFF_EPOCHS_CNT_WEIGHT = 1 / 50


@beartype
def make_cmd_parser():
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
        "--message",
        type=str,
        default="",
        help="the message of checkpoint",
    )

    save_parser.add_argument(
        "--full",
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
        "--diff_time_weight",
        type=int,
        default=DEFAULT_DIFF_TIME_WEIGHT,
        help="",
    )

    train_parser.add_argument(
        "--diff_epochs_cnt_weight",
        type=int,
        default=DEFAULT_DIFF_EPOCHS_CNT_WEIGHT,
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

    call_parser = subparser.add_parser(
        "call",
    )

    call_parser.add_argument(
        "func_name",
        type=str,
        help="The function name to call.",
    )

    call_parser.add_argument(
        "kwargs",
        nargs=argparse.REMAINDER,
        help="Function keyword arguments.",
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
    def _get_log_db_path(proj_dir: pathlib.Path):
        return proj_dir / "log.db"

    @staticmethod
    def _get_ckpt_data_path(proj_dir: pathlib.Path, id: int):
        return proj_dir / f"ckpt_data_{id}.pth"

    @staticmethod
    def _get_cancel_token_path(proj_dir: pathlib.Path):
        return proj_dir / f"cancel_token.txt"

    def __init__(
        self,
        proj_dir: os.PathLike,
        device: torch.device,
    ):
        self.__proj_dir = pathlib.Path(proj_dir)

        if not self.__proj_dir.exists():
            os.makedirs(self.__proj_dir, exist_ok=True)
        else:
            assert self.__proj_dir.is_dir()

        log_db_path = Trainer._get_log_db_path(self.__proj_dir)

        self.__log_db = LogDatabase(log_db_path)

        self.__device = device

        self.__prv: int = -1
        self.__epochs_cnt: int = 0
        self.__avg_loss: float = 0.0

        self.__diff_epochs_cnt = 0

        self.training_core: TrainingCore = None

        utils.write_file(Trainer._get_cancel_token_path(
            self.__proj_dir), "w", "")

    def get_training_core(self):
        return self.training_core

    def set_training_core(self, training_core: TrainingCore):
        self.training_core = training_core

    def show(self):
        print(f"")
        print(f"            prv = {self.__prv}")
        print(f"     epochs cnt = {self.__epochs_cnt}")
        print(f"diff epochs cnt = {self.__diff_epochs_cnt}")
        print(f"       avg loss = {self.__avg_loss}")
        print(f"")

    def get_latest_ckpt_meta(self):
        return self.__log_db.select_latest_ckpt_meta()

    def load(self, id: int):
        assert self.training_core is not None

        ckpt_meta = self.__log_db.select_ckpt_meta({"id": id})

        assert ckpt_meta is not None

        ckpt_data_path = Trainer._get_ckpt_data_path(
            self.__proj_dir, ckpt_meta.id)

        d = torch.load(ckpt_data_path)

        if ckpt_meta.full:
            for field_name in {"module", "optimizer", "scheduler"}:
                if field_name not in d:
                    continue

                val = d[field_name]

                if field_name == "module":
                    val = val.to(self.__device)

                self.training_core.set(field_name, val)
        else:
            for field_name in {"module", "optimizer", "scheduler"}:
                if field_name not in d:
                    continue

                field_val = self.training_core.get(field_name)

                if field_val is not None:
                    field_val.load_state_dict(d[field_name])

                if field_name == "module":
                    field_val.to(self.__device)

        self.__prv = ckpt_meta.prv
        self.__epochs_cnt = ckpt_meta.epochs_cnt
        self.__diff_epochs_cnt = 0
        self.__avg_loss = ckpt_meta.avg_loss

    def load_latest(self):
        ckpt_meta = self.get_latest_ckpt_meta()

        if ckpt_meta is not None:
            self.load(ckpt_meta.id)

    def save(
        self,
        message: str = None,
        full: bool = False,
    ):
        assert self.training_core is not None

        id = int(time.time())

        print(f"saving id={id}")

        self.__log_db.insert_ckpt_meta(CheckpointMeta(
            id=id,
            prv=self.__prv,
            epochs_cnt=self.__epochs_cnt,
            message=message,
            full=full,
            avg_loss=self.__avg_loss,
        ))

        self.__prv = id
        self.__diff_epochs_cnt = 0

        d = dict()

        if full:
            print(f"full saved")

            for field_name in {"module", "optimizer", "scheduler"}:
                d[field_name] = self.training_core.get(field_name)
        else:
            print(f"     saved")

            for field_name in {"module", "optimizer", "scheduler"}:
                field_val = self.training_core.get(field_name)

                d[field_name] = None if field_val is None else \
                    field_val.state_dict()

        print(f"{d=}")

        torch.save(
            d, Trainer._get_ckpt_data_path(self.__proj_dir, id))

        print(f"saved")

    def _train(self):
        training_result = self.training_core.train()

        self.__epochs_cnt += 1
        self.__diff_epochs_cnt += 1
        self.__avg_loss = training_result.avg_loss

    def train(
        self,
        epochs_cnt: int = 1,
        diff_time_weight: float = DEFAULT_DIFF_TIME_WEIGHT,
        diff_epochs_cnt_weight: float = DEFAULT_DIFF_EPOCHS_CNT_WEIGHT,
    ):
        assert 0 <= epochs_cnt

        assert 0 <= diff_time_weight
        assert 0 <= diff_epochs_cnt_weight

        cancel_token_path = Trainer._get_cancel_token_path(self.__proj_dir)

        utils.write_file(cancel_token_path, "w", "")

        for _ in range(epochs_cnt):
            self._train()

            cur_time = int(time.time())

            diff_time = cur_time - self.__prv

            print(f"{diff_time=}")
            print(f"{self.__diff_epochs_cnt=}")

            if 1 - 1e-3 <= diff_time_weight * diff_time + diff_epochs_cnt_weight * self.__diff_epochs_cnt:
                print(f"auto save triggered")
                self.save()

            if len(utils.read_file(cancel_token_path, "r")) != 0:
                break

    def eval(self):
        self.training_core.eval()

    def call(self, func_name: str, kwargs: dict[str, object]):
        getattr(self.training_core, func_name)(**kwargs)

    def _cli_handler(self, cmd_parser):
        cmd = shlex.split(input("trainer> "))

        args = cmd_parser.parse_args(cmd)

        match args.op:
            case "show":
                self.show()

            case "load":
                self.load(args.id)

            case "load_latest":
                self.load_latest()

            case "save":
                self.save(args.message, args.full)

            case "train":
                self.train(
                    epochs_cnt=args.epochs_cnt,
                    diff_time_weight=args.diff_time_weight,
                    diff_epochs_cnt_weight=args.diff_epochs_cnt_weight,
                )

            case "eval":
                self.eval()

            case "call":
                kwargs: dict[str, object] = dict()

                for kwarg in args.kwargs:
                    kwarg: str

                    if "=" in kwarg:
                        key, value = kwarg.split("=", 1)
                        kwargs[key.strip().strip("-")] = eval(value)

                self.call(args.func_name, kwargs)

            case "exit":
                return False

        return True

    def enter_cli(self):
        cmd_parser = make_cmd_parser()

        while True:
            try:
                if not self._cli_handler(cmd_parser):
                    return
            except KeyboardInterrupt:
                print(traceback.format_exc())
                sys.exit(0)
                break
            except:
                print(traceback.format_exc())
                continue
