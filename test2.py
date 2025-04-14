import shlex
import prompt_toolkit


def main2():
    prompt_session = prompt_toolkit.PromptSession(
        "trainer> ",
        style=prompt_toolkit.styles.Style.from_dict({
            "prompt": "ansigreen bold",
            "input": "ansiblue",
            "": "ansiyellow",
        }),
    )

    while True:
        cmd = prompt_session.prompt()

        print(f"{cmd=}")


if __name__ == "__main__":
    main2()
