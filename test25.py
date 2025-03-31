import argparse
import sys
import shlex


def main1():

    # ---

    cmd = input("traine> ")

    arg = parser.parse_args(shlex.split(cmd))

    print(arg)


if __name__ == "__main__":
    main1()
