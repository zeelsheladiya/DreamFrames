import time
from rich.console import Console
from utils.video_utils import VideoOperation
from utils.llm_utils import AiArtOperation

def main():
    console = Console()
    video_operation = VideoOperation()
    ai_operation = AiArtOperation()

    with console.status("[bold green]Script in process...") as status:
        while True:
            time.sleep(1)
            console.print("1. Script started!", style="bold blue")
            console.print("2. Video Operation Started: Extracting Frames form Video!", style="bold blue")
            video_operation.extract_frames()
            console.print("3. Video Operation Finished: Extracting Frames Completed!", style="bold blue")
            time.sleep(1)
            console.print("4. AI Operation Started: Converting to AI Art!", style="bold blue")
            ai_operation.convert_frames_to_ai_art()
            console.print("5. AI Operation Finished: Complete Converting to AI Art!", style="bold blue")

            break


if __name__ == "__main__":
    main()