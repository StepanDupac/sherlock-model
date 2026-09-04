import functools
import http.server
import os
import socketserver
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = 'explainer.html'


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass


def main():
    page = os.path.join(HERE, PAGE)
    if not os.path.isfile(page):
        sys.exit(f'{PAGE} not found in {HERE}')

    handler = functools.partial(Handler, directory=HERE)
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(('127.0.0.1', 0), handler) as server:
        url = f'http://127.0.0.1:{server.server_address[1]}/{PAGE}'
        size = os.path.getsize(page) / 1e6
        print(f'Sherlock explainer  ({size:.1f} MB)')
        print(f'  {url}')
        print('  Ctrl+C to stop')
        threading.Timer(0.6, webbrowser.open, [url]).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nstopped')


if __name__ == '__main__':
    main()
