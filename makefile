fix:
	black --line-length 100 src
	pylint src

data/cache: data
	mkdir data/cache

data/raw: data
	-mkdir data/raw

data:
	mkdir data


external: external/Unsupervised-Aspect-Extraction 

# This is commented out because it's outdated (original author's version, but
# python 2
#external/Unsupervised-Aspect-Extraction:
	#git clone https://github.com/ruidan/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction

external/Unsupervised-Aspect-Extraction:
	git clone https://github.com/harpaj/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction
	virtualenv -p python3.6 external/Unsupervised-Aspect-Extraction
	cd external/Unsupervised-Aspect-Extraction && \
		. bin/activate && \
		pip install -r requirements.txt


datalocs: dataloc/kaggle1 dataloc/webhose_political

dataloc/kaggle1:
	firefox https://www.kaggle.com/snapcrack/all-the-news

	
getdata: data/raw/kaggle1
	
data/raw/kaggle1: data/raw
	cp ~/Downloads/all-the-news.zip ./data/raw/all-the-news.zip
	unzip ./data/raw/all-the-news.zip -d data/raw/kaggle1
	rm data/raw/all-the-news.zip


dataloc/webhose_political:
	firefox https://webhose.io/free-datasets/political-news-articles/


data/raw/webhose_political: data/raw
	cp ~/Downloads/660_20170904095215.zip ./data/raw/webhose_political.zip
	unzip ./data/raw/webhose_political.zip -d data/raw/webhose_political
	rm data/raw/webhose_political.zip
	unzip data/raw/webhose_political/660_webhose-2015-10-new_20170904095249.zip -d data/raw/webhose_political
	rm data/raw/webhose_political/660_webhose-2015-10-new_20170904095249.zip


env:
	virtualenv -p python3.6 src/aspect_detection/neural_attn_model
	cd src/aspect_detection/neural_attn_model && \
		. bin/activate && \
		pip install -r requirements.txt
	

.PHONY: external getdata dataloc/kaggle1 datalocs fix env dataloc/webhose_political
