
import utility
import sys
import logging


def test():
    logging.info("Hello world from the test function")

if __name__ == "__main__":
    utility.init_logging(sys.argv[1])
    test()

    #ARGS = parse()
    #docify(ARGS.input_folder, ARGS.output_path, ARGS.count, ARGS.column, ARGS.overwrite)
