external: external/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction-Py3

external/Unsupervised-Aspect-Extraction:
	git clone https://github.com/ruidan/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction

external/Unsupervised-Aspect-Extraction-Py3:
	git clone https://github.com/harpaj/Unsupervised-Aspect-Extraction external/Unsupervised-Aspect-Extraction-Py3
	virtualenv -p python3.6 external/Unsupervised-Aspect-Extraction-Py3

.PHONY: external
