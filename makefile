data/raw: data
	mkdir data/raw

data:
	mkdir data


external: external/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction-Py3

external/Unsupervised-Aspect-Extraction:
	git clone https://github.com/ruidan/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction

external/Unsupervised-Aspect-Extraction-Py3:
	git clone https://github.com/harpaj/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction-Py3
	virtualenv -p python3.6 external/Unsupervised-Aspect-Extraction-Py3


datalocs: dataloc/kaggle1

dataloc/kaggle1:
	firefox https://www.kaggle.com/snapcrack/all-the-news

	
getdata: data/raw/kaggle1
	
data/raw/kaggle1: data/raw
	cp ~/Downloads/all-the-news.zip ./data/raw/all-the-news.zip
	unzip ./data/raw/all-the-news.zip -d data/raw/kaggle1
	rm data/raw/all-the-news.zip

.PHONY: external getdata dataloc/kaggle1 datalocs
