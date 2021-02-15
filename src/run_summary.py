import os
import subprocess
import sys

def run_presumm(input_path, result_path):
    """
    Python client for PreSumm

    Parameters:
    input_path (str): The folder path where pdfs are.
    result_path (str): The folder path where generated tables will be.
    """

    input_path = input_path
    result_path = result_path
    company_folders = ['MMM', 'AXP', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'PG', 'TRV', 'VZ', 'WMT', 'DIS']
#     company_folders = ['MMM', 'AXP', 'BA', 'CAT', 'CVX', 'CSCO', 'KO']
#     company_folders = ['HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK']
#     company_folders = ['MSFT', 'PG', 'TRV', 'VZ', 'WMT', 'DIS']

    pid = str(os.getpid())
    f = open(f'{company_folders[0]}_pid', 'w')
    f.write(pid)
    f.close()
    for cf in company_folders:
        year_folders = os.listdir(f'{input_path}/{cf}')
        for yf in year_folders:
            if not os.path.exists(f'{result_path}/{cf}/{yf}'):
                os.makedirs(f'{result_path}/{cf}/{yf}')
            filenames = [f for f in os.listdir(f'{input_path}/{cf}/{yf}') if f.endswith('.txt')]
            for filename in filenames:
                input_path_s = os.path.expanduser(input_path) + f'/{cf}/{yf}/' + filename
                result_path_s = os.path.expanduser(result_path) + f'/{cf}/{yf}/' + filename

                command = f'python3 train.py -task ext -mode test_text -test_from ../models/bert_transformer/bertext_cnndm_transformer.pt -text_src {input_path_s} -result_path {result_path_s} -visible_gpus 0,1,2 -max_pos 512'

                output, error = subprocess.Popen([command],
                                                shell=True, universal_newlines=True).communicate()
                print('finish summarizing %s to %s' % (input_path, output_path))

#     filenames = [f for f in os.listdir(input_path) if f.endswith('.txt')]

#     for filename in filenames:
#         input_path_s = os.path.expanduser(input_path) + '/' + filename
#         result_path_s = os.path.expanduser(result_path) + '/' + filename

#         command = 'python3 train.py -task ext -mode test_text -test_from ../models/bert_transformer/bertext_cnndm_transformer.pt -text_src {} -result_path {} -visible_gpus -1 -max_pos 512'.format(input_path_s, result_path_s)

#         output, error = subprocess.Popen([command],
#                                         shell=True, universal_newlines=True).communicate()
#         print('finish summarizing %s to %s' % (input_path, output_path))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    run_presumm(input_path, output_path)

    #-max_pos 512 -max_length 200 -alpha 0.95 -min_length 50