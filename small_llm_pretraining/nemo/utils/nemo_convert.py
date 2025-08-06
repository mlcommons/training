if __name__ == "__main__":
    import argparse
    from nemo.collections.llm.gpt.model.llama import HFLlamaImporter
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/source", type=str)
    parser.add_argument("--destination", default="/destination", type=str)
    args = parser.parse_args()
    
    importer = HFLlamaImporter(args.source)
    importer.apply(args.destination)
