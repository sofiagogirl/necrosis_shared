import threading, sys

global is_running
is_running = True


class Watcher(object):
    def __init__(self):
        self.thread = threading.Thread(target=self.input)
        self.thread.setDaemon(True)
        self.thread.start()

    def input(self):
        global is_running
        while is_running:
            try:
                a = str(input("Enter 'e' to stop running.\n"))
                if a == 'e':
                    is_running = False
                else:
                    print("Illegal input!")
            except:
                print('Error encountered.')

    def check_stop(self):
        self.thread.join(timeout=0.0000001)
        if not is_running:
            print('Program interrupted by user.')
            sys.exit()

