import argparse
import sys

import threading
import timedinput


class TimeoutException(Exception):
    pass


def TimedInput(prompt, timeout):
    def TimeoutHandler():
        raise TimeoutException()

    timer = threading.Timer(timeout, TimeoutHandler)
    timer.start()

    try:
        user_input = input(prompt)
        timer.cancel()
    except TimeoutException as e:
        raise e

    return user_input


def main1():
    try:
        input = timedinput.timedinput(
            "input anything to interrupt training: ", 10)

        print(f"training are interrupted")
    except timedinput.TimeoutOccurred:
        pass

    print(f"continue next epoch")


if __name__ == "__main__":
    main1()
