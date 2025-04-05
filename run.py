
from classifier import *

if __name__ == "__main__":
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.txt')

    def_categ = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]

    raw = config.getboolean('SETUP', 'raw')
    dir = config.getboolean('SETUP', 'dir')
    pro = config.getboolean('SETUP', 'pro')
    inner = config.getboolean('SETUP', 'inner')
    credibility = config.getfloat('SETUP', 'credibility')

    test_dir = config.get('INPUT', 'FOLDER_INPUT')

    # cur = Path.cwd()  # directory with this script
    cur = Path(__file__).resolve().parent  # directory with this script
    output_dir = Path(config.get('OUTPUT', 'FOLDER_RESULTS'))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    load_dotenv()

    parser = argparse.ArgumentParser(description='Page sorter based on ViT')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-c', "--cred", type=float, default=credibility, help="Minimal credibility of DeepDoctection predictions")
    parser.add_argument("--dir", default=dir, help="Process whole directory (if -d not used)", action="store_true")
    parser.add_argument("--inner", help="Process subdirectories of the given directory as well (FALSE by default)", default=inner, action="store_true")
    parser.add_argument("--raw", help="Output raw scores for all categories", default=raw, action="store_true")
    parser.add_argument("--pro", help="Use PyTesseract", default=pro, action="store_true")

    args = parser.parse_args()

    input_dir = Path(test_dir) if args.directory is None else Path(args.directory)
    raw = args.raw

    # locally creating new directory paths instead of context.txt variables loaded with mistakes
    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
        os.makedirs(f"{output_dir}/plots")
        os.makedirs(f"{output_dir}/layouts_dd")
        os.makedirs(f"{output_dir}/layout_stat")
        os.makedirs(f"{output_dir}/pages")


    categories = def_categ
    print(f"Category input directories found: {categories}")

    # Initialize the classifier
    classifier = ImageClassifier(pro=args.pro, num_labels=len(categories), credibility=args.cred, out_folder=output_dir)

    # classifier.load_model(str(model_path))

    # if args.file is not None:
    #     # pred_scores = classifier.top_n_predictions(args.file, top_N)
    #
    #     labels = [categories[i[0]] for i in pred_scores]
    #     scores = [round(i[1], 3) for i in pred_scores]
    #
    #     print(f"File {args.file} predicted:")
    #     for lab, sc in zip(labels, scores):
    #         print(f"\t{lab}:  {round(sc * 100, 2)}%")

    if args.dir or args.directory is not None:

        if args.inner:
            test_images = sorted(directory_scraper(Path(test_dir), "png"))
        else:
            test_images = sorted(os.listdir(test_dir))
            test_images = [os.path.join(test_dir, img) for img in test_images]

        # classifier.pdf_to_json(input_dir)
        df = classifier.post_process()


        df.to_csv(f"{output_dir}/tables/{time_stamp}_pdf_summary.csv", sep=",", index=False)


        # test_loader = classifier.create_dataloader(test_images, batch)
        #
        # test_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)
        #
        # rdf, raw_df = dataframe_results(test_images,
        #                                 test_predictions,
        #                                 categories,
        #                                 top_N,
        #                                 raw_prediction)
        #
        # rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
        # rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_TOP-{top_N}.csv", sep=",", index=False)
        # print(f"Results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")
        #
        # if raw:
        #     raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
        #     raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_RAW.csv", sep=",", index=False)
        #     print(f"RAW Results are recorded into {output_dir}/tables/ directory")


