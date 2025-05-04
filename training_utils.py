import argparse
import collections
import dataclasses
import datetime
import math
import os
import pathlib
import shlex
import sqlite3
import sys
import time
import traceback
import typing

import prompt_toolkit
import tabulate
import torch
from beartype import beartype

from . import sqlite_utils, utils


@beartype
@dataclasses.dataclass
class CheckpointMeta:
    id: int
    prv: int
    epoch: int
    time: datetime.datetime
    message: typing.Optional[str]
    training_message: typing.Optional[str]
    full: bool


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

                    epoch INT,
                    time TEXT,

                    message TEXT,
                    training_message TEXT,

                    full INT NOT NULL,

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
                "epoch": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "time": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: utils.serialize_datetime(x, "SEC"),
                    deserialize=lambda x: utils.deserialize_datetime(x),),
                "message": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "training_message": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: x,
                    deserialize=lambda x: x,),
                "full": sqlite_utils.DataTableColAttr(
                    serialize=lambda x: 1 if x else 0,
                    deserialize=lambda x: x != 0,),
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
    message: str


@beartype
class TrainerCore:
    def get_epoch(self) -> int:
        raise NotImplementedError()

    # ---

    def state_dict(self, full: bool = False)  \
            -> collections.OrderedDict[str, typing.Any]:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: typing.Mapping[str, object]):
        raise NotImplementedError()

    # ---

    def train(self) -> TrainingResult:
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


DEFAULT_DIFF_TIME_WEIGHT = 1 / (30 * 60)
DEFAULT_DIFF_EPOCHS_CNT_WEIGHT = 1 / 50


@beartype
def make_cmd_parser():
    parser = argparse.ArgumentParser(prog="parser")

    subparser = parser.add_subparsers(
        dest="op", required=True)

    # ---

    nop_parser = subparser.add_parser(
        "nop", help="nop")

    # ---

    show_parser = subparser.add_parser(
        "show", help="show status")

    # ---

    load_parser = subparser.add_parser(
        "load", help="load a checkpoint")

    load_parser.add_argument(
        "--id", type=int, required=True, help="the target checkpoint's id")

    # ---

    load_latest_parser = subparser.add_parser(
        "load_latest", help="load the latest checkpoint")

    # ---

    save_parser = subparser.add_parser(
        "save", help="save current state to a checkpoint")

    save_parser.add_argument(
        "--message", type=str, default="", help="the message of checkpoint")

    save_parser.add_argument(
        "--full", action="store_true", help="save the total state")

    # ---

    train_parser = subparser.add_parser(
        "train", help="train the module")

    train_parser.add_argument(
        "--epochs_cnt",
        type=int, default=1, help="the number of epochs to train")

    train_parser.add_argument(
        "--diff_epoch_weight", type=int, default=DEFAULT_DIFF_EPOCHS_CNT_WEIGHT)

    train_parser.add_argument(
        "--diff_time_weight", type=int, default=DEFAULT_DIFF_TIME_WEIGHT)

    train_parser.add_argument(
        "--loop", action="store_true")

    # ---

    eval_parser = subparser.add_parser(
        "eval", help="eval the module")

    # ---

    call_parser = subparser.add_parser(
        "call")

    call_parser.add_argument(
        "func_name", type=str, help="The function name to call.")

    call_parser.add_argument(
        "kwargs", nargs=argparse.REMAINDER, help="Function keyword arguments.")

    # ---

    exe_cmd_file_parser = subparser.add_parser(
        "exe_cmd_file", help="execute file commands")

    # ---

    exit_parser = subparser.add_parser(
        "exit", help="exit the cli mode")

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

    def __init__(self, proj_dir: os.PathLike, trainer_core: TrainerCore):
        self.__proj_dir = utils.to_pathlib_path(proj_dir)

        if not self.__proj_dir.exists():
            self.__proj_dir.mkdir(parents=True, exist_ok=True)
        else:
            assert self.__proj_dir.is_dir()

        log_db_path = Trainer._get_log_db_path(self.__proj_dir)

        self.__log_db = LogDatabase(log_db_path)

        self.__prv_ckpt: CheckpointMeta = None
        self.__cur_time: datetime.datetime = datetime.datetime.now()

        self.__training_result: TrainingResult = None

        self.trainer_core = trainer_core

        utils.write_file(Trainer._get_cancel_token_path(
            self.__proj_dir), "w", "")

        self.prompt_session = prompt_toolkit.PromptSession(
            "trainer> ",
            style=prompt_toolkit.styles.Style.from_dict({
                "prompt": "ansigreen bold",
                "input": "ansiblue",
                "": "ansiyellow",
            }),
        )

    def show(self) -> None:
        cur_epoch = self.trainer_core.get_epoch()

        if self.__prv_ckpt is None:
            prv_ckpt_id = -1
            prv_ckpt_epoch = -1
            prv_ckpt_time = None

            diff_epoch = None
            diff_time = None
        else:
            prv_ckpt_id = self.__prv_ckpt.id
            prv_ckpt_epoch = self.__prv_ckpt.epoch
            prv_ckpt_time = self.__prv_ckpt.time

            diff_epoch = cur_epoch - prv_ckpt_epoch
            diff_time = self.__cur_time - prv_ckpt_time

        training_message = self.__training_result.message if self.__training_result is not None else ""

        print(tabulate.tabulate([
            ("prv ckpt id", prv_ckpt_id),
            ("prv ckpt epoch", prv_ckpt_epoch),
            ("cur epoch", self.trainer_core.get_epoch()),
            ("cur time", utils.serialize_datetime(self.__cur_time, "SEC")),
            ("traininer message", training_message),
            ("diff epoch", diff_epoch),
            ("diff time", diff_time),
        ]))

    def get_latest_ckpt_meta(self) -> CheckpointMeta:
        return self.__log_db.select_latest_ckpt_meta()

    def load(self, id: int) -> None:
        assert self.trainer_core is not None

        ckpt_meta = self.__log_db.select_ckpt_meta({"id": id})

        assert ckpt_meta is not None

        ckpt_data_path = Trainer._get_ckpt_data_path(
            self.__proj_dir, ckpt_meta.id)

        self.trainer_core.load_state_dict(torch.load(ckpt_data_path))

        self.__prv_ckpt = ckpt_meta

    def load_latest(self) -> None:
        ckpt_meta = self.get_latest_ckpt_meta()

        if ckpt_meta is not None:
            self.load(ckpt_meta.id)

    def save(
        self,
        message: str = None,
        full: bool = False,
    ) -> None:
        assert self.trainer_core is not None

        id = int(time.time())

        print(f"saving id={id}")

        ckpt = CheckpointMeta(
            id=id,
            prv=-1 if self.__prv_ckpt is None else self.__prv_ckpt.id,
            epoch=self.trainer_core.get_epoch(),
            time=self.__cur_time,
            message=message,
            training_message=None if self.__training_result is None
            else self.__training_result.message,
            full=full,
        )

        self.__log_db.insert_ckpt_meta(ckpt)

        self.__prv_ckpt = ckpt

        d = None

        if full:
            print(f"full saved")
            d = self.trainer_core.state_dict(True)
        else:
            print(f"     saved")
            d = self.trainer_core.state_dict(False)

        torch.save(d, Trainer._get_ckpt_data_path(self.__proj_dir, id))

        print(f"saved")

    def train(
        self,
        epochs_cnt: int = 1,
        diff_epoch_weight: float = DEFAULT_DIFF_EPOCHS_CNT_WEIGHT,
        diff_time_weight: float = DEFAULT_DIFF_TIME_WEIGHT,
    ) -> None:
        assert 0 <= epochs_cnt

        assert 0 <= diff_epoch_weight
        assert 0 <= diff_time_weight

        cancel_token_path = Trainer._get_cancel_token_path(self.__proj_dir)

        utils.write_file(cancel_token_path, "w", "")

        for _ in range(epochs_cnt):
            self.__training_result = self.trainer_core.train()

            epoch = self.trainer_core.get_epoch()
            self.__cur_time = datetime.datetime.now()

            if self.__prv_ckpt is None:
                diff_epoch = math.inf
                diff_time = math.inf

                do_auto_save = True
            else:
                prv_ckpt_epoch = self.__prv_ckpt.epoch

                diff_epoch = epoch - prv_ckpt_epoch
                diff_time = self.__cur_time - self.__prv_ckpt.time

                do_auto_save = 1 - 1e-3 <= (
                    diff_epoch_weight * diff_epoch +
                    diff_time_weight * diff_time.total_seconds()
                )

            print(f"{diff_epoch=}")
            print(f"{diff_time=}")

            if do_auto_save:
                print(f"auto save triggered")
                self.save()

            if len(utils.read_file(cancel_token_path, "r")) != 0:
                break

    def call(self, func_name: str, kwargs: dict[str, object]) -> None:
        ret = getattr(self.trainer_core, func_name)(**kwargs)
        print(f"{ret=}")

    def _cli_handler(self, cmd_parser) -> bool:
        cmd = shlex.split(self.prompt_session.prompt())

        if len(cmd) == 0:
            return True

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
                    diff_epoch_weight=args.diff_epoch_weight,
                    diff_time_weight=args.diff_time_weight,
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

    def enter_cli(self) -> None:
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
