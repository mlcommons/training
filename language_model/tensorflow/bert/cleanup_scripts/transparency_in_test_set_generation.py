import glob

output_filename = 'wiki_test_set.txt'

test_articles = []

file_glob = glob.glob('./results/part*', recursive=False)

with open(output_filename, mode='w', newline='\n') as ofile:
  for filename in file_glob:
    articles_in_file = []
    with open(filename, mode='r', newline='\n') as ifile:
      lines = ifile.read()
      articles_in_file_tmp = lines.split('\n\n')
      articles_in_file = []
      for item in articles_in_file_tmp:
        if item.rstrip() != '':
          articles_in_file.append(item)
    
    target_article = min(42, len(articles_in_file) // 2)
    test_articles.append(articles_in_file[target_article])

    with open(filename, mode='w', newline='\n') as ifile:
      for article in articles_in_file[:target_article]:
        ifile.write(article)
        ifile.write('\n\n')

      for article in articles_in_file[target_article+1:]:
        ifile.write(article)
        ifile.write('\n\n')

  for article in test_articles:
    ofile.write(article)
    ofile.write('\n\n')

print("n_articles =", len(test_articles))
