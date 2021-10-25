import mkmd
import argparse as arg_parser
import numpy as np


def main():
    parser = arg_parser.ArgumentParser(description='Optional app description')
    app_names = mkmd.application_names.values()
    names = ''
    for app_name in app_names:
        names += app_name + ','

    parser.add_argument('--mkmd_name', type=str, help='Select one application name: ' + 'ALL,' + names)
    parser.add_argument('--partition_subkernel_on', dest='partition', action='store_true')
    parser.add_argument('--partition_subkernel_off', dest='partition', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    app_name = args.mkmd_name
    use_subkernel_mkmd = args.partition

    selected_platform = mkmd.Platforms.PLATFORM_A
    matrix_width = 4096

    threshold_offload = 1 / 100.0  # offload in %
    show_gantt_graph = True

    if app_name == 'ALL':

        for key, application_name in mkmd.application_names.items():
            setup = mkmd.MKMDSetup(application_name, matrix_width, threshold_offload,
                                   selected_platform, use_subkernel_mkmd, show_gantt_graph)

            mkmd.call_mkmd(setup)
    else:

        if app_name is not None and any(app_name in s for s in app_names):
            setup = mkmd.MKMDSetup(app_name, matrix_width, threshold_offload,
                                   selected_platform, use_subkernel_mkmd, show_gantt_graph)
            mkmd.call_mkmd(setup)
        else:
            print("Unsupported application name, please check: ./app -h , EXIT!")
            return -1

    return 0


if __name__ == '__main__':
    main()
