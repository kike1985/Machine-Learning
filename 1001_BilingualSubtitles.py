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

    in_path = f'{base_path}/in/{path}'
    out_path = f'{base_path}/out/{path}'

    file_in = open(in_path, 'r')
    text_in = file_in.readlines()
    file_in.close()

    text_out, new_lines = [], 0

    bar = progress.add_task(f'[red]thread {threading.get_ident()}...', total=len(text_in))

    for i, text in enumerate(text_in):
        if bool(re.match(time_pattern, text)) == True and len(text_out) > 1:
            for j in range(3, 10):
                k = i-j

                time_before = text_in[k]

                if bool(re.match(time_pattern, time_before)) == True:
                    no_before = text_in[k-1]

                    text_out.insert(i+new_lines-1, no_before)      # Se agrega el numero anterior
                    new_lines += 1
                    text_out.insert(i+new_lines-1, time_before)    # Se agrega el tiempo anterior
                    new_lines += 1

                    # Se agrega el texto anterior que puede ser de varias lineas
                    t_text = ''.join([text_in[k+l] for l in range(1, j-1)])

                    try:
                        t_text = '{\\an8}' + translator.translate(t_text) + '\n\n'
                    except Exception as e:
                        progress.remove_task(bar)
                        return

                    text_out.insert(i+new_lines-1, t_text)
                    new_lines += 1

                    break

        text_out.append(text)
        progress.update(bar, advance=1)

    progress.remove_task(bar)

    file_out = open(out_path, 'w')
    file_out.writelines(text_out)
    file_out.close()


with ThreadPoolExecutor(max_workers=6) as executor:
    progress = Progress()
    progress.start()

    futures = [executor.submit(build_thread, path, progress)
               for path in os.listdir(f'{base_path}/in')]

    executor.shutdown(wait=True)
    progress.stop()
    print('Finish!!!')
