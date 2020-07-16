import os
import subprocess
import sys

def run_presumm(input_path, result_path):
    """
    Python client for PreSumm

    Parameters:
    input_path (str): The folder path where txts are.
    result_path (str): The folder path where generated summary will be.
    """

    filenames = [f for f in os.listdir(input_path) if f.endswith('.txt')]

    for filename in filenames:
        input_path_s = os.path.expanduser(input_path) + '/' + filename
        result_path_s = os.path.expanduser(result_path) + '/' + filename

        command = 'python3 train.py -task ext -mode test_text -test_from ../models/bert_transformer/bertext_cnndm_transformer.pt -text_src {} -result_path {} -visible_gpus -1'.format(input_path_s, result_path_s)

        output, error = subprocess.Popen([command],
                                        shell=True, universal_newlines=True).communicate()
        print('finish summarizing %s to %s' % (input_path, output_path))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    run_presumm(input_path, output_path)