# LEVEN Dataset Format

Each `.jsonl` file is a subset of LEVEN and each line in the files is a json string for a document.  

For the `train.jsonl` and `valid.jsonl` the json format is as below:

```json
{
    "title": "李某波犯猥亵儿童罪一审刑事判决书",    #the tiltle of the document
    "id": "a6f3b705d93e441dbd3a29365e854193",  #an unique string for each document
    "crime": "猥亵儿童罪",                      #the related crime of the document 
    "case_no": "（2014）梅兴法刑初字第344号",     #the case number of the document
    "content": [ #the content of the document. A list, each item is a dict for a sentence
    		{
    		 "sentence":"...", #a string, the plain text of the sentence
    		 "tokens": ["...", "..."] #a list, tokens of the sentence
			}
	],
	"events":[ #a list for annotated events, each item is a dict for an event
        {
            "id": '0fd7970c76d64c5d9ac1c015609c028b', #an unique string for the event
            "type": '租用/借用',                       #the event type
            "type_id": 22,                           #the numerical id for the event type
            "mention":[ #a list for the event mentions of the event, each item is a dict
            	{
              		"trigger_word": "租住", #a string of the trigger word or phrase
              		"sent_id": 0, # the index of the corresponding sentence, strates with 0
              		"offset": [41, 42],# the offset of the trigger word in the tokens list
					"id": "2db165c25298aefb682cba50c9327e4f", # an unique string for the event mention
              	}
             ]
        }
    ],
	"negative_triggers":[#a list for negative instances, each item is a dict for a negative mention
        {
            "trigger_word": "出生",
            "sent_id": 0,
            "offset": [21, 22],
			"id": "66571b43dcf9461cb7ce979875fc9287",
        }
    ]
}
```

For the `test.jsonl`, the format is almost the same but we hide the annotation results:
Please refer to https://github.com/thunlp/LEVEN for the evaluation method.

```json
{
    "title": "姚均飞强奸罪一审刑事判决书",            #the tiltle of the document
    "id": "9720823b46ea4efebb52539f2016d8b8",     #an unique string for each document
    "crime": "强奸罪",                            #the related crime of the document
    "case_no": "（2018）渝0154刑初280号",           #the case number of the document
    "content": [ #the content of the document. A list, each item is a dict for a sentence
    		{
    		 "sentence":"...", #a string, the plain text of the sentence
    		 "tokens": ["...", "..."] #a list, tokens of the sentence
			}
	],
	"candidates":[ #a list for trigger candidiates, each item is a dict for a trigger or a negative instance, you need to classify the type for each candidate
        {
            "trigger_word": "认识",
            "sent_id": 0,
            "offset": [28, 29],
			"id": "f3f93191743a4c63966f5c48f8f6383c",
        }
    ]
}
```