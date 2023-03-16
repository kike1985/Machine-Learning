import os
import re
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress
import threading


time_pattern = "[0-9]{2}\:[0-9]{2}\:[0-9]{2}\,[0-9]{3}\ --> [0-9]{2}\:[0-9]{2}\:[0-9]{2}\,[0-9]{3}"

base_path = 'Notebooks/1001_BilingualSubtitles/resources'


def build_thread(path, progress):
    if path.split('.')[-1] != 'srt' or os.path.exists(f'{base_path}/out/{path}'):
        return

    translator = GoogleTranslator(source='en', target='es')

    with open(f'{base_path}/in/{path}', 'r') as file_in:
        text_in = file_in.readlines()

    text_out, new_lines = [], 0

    bar = progress.add_task(f'[red]thread {threading.get_ident()} [{len(text_in)} items]', total=len(text_in))

    for i, text in enumerate(text_in):
        if bool(re.match(time_pattern, text)) == True and len(text_out) > 1:
            for j in range(3, 10):
                k = i-j

                if bool(re.match(time_pattern, text_in[k])) == True:
                    t_text = ''.join([text_in[k+l] for l in range(1, j-1)])

                    try:
                        t_text = '<font color="#D900D9">' + translator.translate(t_text) + '</font>\n'
                    except Exception as e:
                        progress.remove_task(bar)
                        raise e
                    else:
                        text_out.insert(i+new_lines-2, t_text)
                        new_lines += 1
                        break

        text_out.append(text)
        progress.update(bar, advance=1)

    with open(f'{base_path}/out/{path}', 'w', encoding='utf-8') as file_out:
        file_out.writelines(text_out)

    progress.remove_task(bar)


with ThreadPoolExecutor(max_workers=6) as executor:
    progress = Progress()
    progress.start()

    futures = [executor.submit(build_thread, path, progress)
               for path in os.listdir(f'{base_path}/in')]

    executor.shutdown(wait=True)
    progress.stop()

    print('Errors:', [e.exception() for e in futures if e.exception() != None])
    print('Finish!!!')
