import sys
import threading

is_running = True


class Watcher:
    def __init__(self):
        self.thread = threading.Thread(target=self._listen_for_input)
        self.thread.daemon = True
        self.thread.start()

    def _listen_for_input(self):
        global is_running
        while is_running:
            try:
                user_input = input("Enter 'e' to stop running.\n")
                if user_input.strip() == 'e':
                    is_running = False
                else:
                    print("Invalid input. Enter 'e' to stop.")
            except Exception:
                print('Error encountered.')

    def check_stop(self):
        self.thread.join(timeout=1e-7)
        if not is_running:
            print('Program interrupted by user.')
            sys.exit()