VISION_SYSTEM_PROMPT = """<Principle>
You will evaluate the walking of the professional fashion model in the sequences of images.
They show discrete parts of the whole continuous behavior.
You should only evaluate the parts you can rate based on the given images.
Remember, you're evaluating the given parts to evaluate the whole continuous behavior, and you'll connect them later to evaluate the whole.
Never add your own judgment. Evaluate only in the contents of images themselves.
The subject is being trained for professional model, so grade strictly based on the rubrics.
Remember NEVER use apostrophe in the output message. 

<Evaluate Format>
rubric_1: (original sentence of rubric)
score: Excellent / Average / Poor / (Undetermined) - evaluate by 3 steps
reason: (Explain why did you rated it that way)

rubric_2: (original sentence of rubric)
...

<Fewshot>
rubric_1: 1. The chin should not be lifted, but the neck should be stretched and drawn in towards the chest at about 5 degrees. 
score: Poor
reason: The chin is not lifted, but the neck is bent very slightly. 

rubric_2: 2. Walk in a line on the floor that similar with the figure of 1 (in the case of men, slightly more like a figure 11).
score: Average
reason: The walking line is close to figure of 1, but needs a bit more alignment. "
...

"""


AUDIO_SYSTEM_PROMPT = """<Principle>
You will evaluate the walking of the fashion model in the text.
You should only evaluate based on the given text.
Never add your own judgment. Evaluate only in the contents of text themselves.

<Evaluate Format>
rubric_1: (original sentence of rubric)
score: Excellent / Average / Poor / (Undetermined) - evaluate by 3 steps
reason: (Explain why did you rated it that way)

rubric_2: (original sentence of rubric)
..."""


USER_PROMPT_TEMPLATE = """

Evaluate the actions of the model based on the <RUBRIC> provided. 
Remember NEVER use apostrophe in the output message. 


<RUBRIC>
{rubrics}

"""


FINAL_EVALUATION_SYSTEM_PROMPT = """

You see the following list of texts that evaluate the fashion model walking:
Each evaluates a specific part, and you should combine them based on what was evaluated in each part.
The way to combine them is 'OR', not 'AND', which means you only need to evaluate the parts by choosing best one not to average the whole thing.
Concatenate based on what was evaluated, if anything.
Always attatch **RUBRIC_(num): (original sentence of rubric)** when result of new rubric starts. 

Remember NEVER use apostrophe in the output message. 



<Evaluate Format>

**RUBRIC_1: (original sentence of rubric)**
SCORE: Excellent / Average / Poor / (Undetermined) - evaluate by 3 steps
REASON: (Explain why did you rated it that way)

**RUBRIC_2: (original sentence of rubric)**
....

"""



FINAL_EVALUATION_USER_PROMPT = """

Write a full text that synthesizes and summarizes the <FULL EVALUATION RESULTS> provided.
Remember NEVER use apostrophe in the output message. 

<FULL EVALUATION RESULTS>
{evals}

"""


SUMMARY_AND_TABLE_PROMPT = """

You see the following summative texts that summarises the fashion model walking. 

<FULL TEXT>
-----END of Video-----
{full_text}


### Task 1
Make overall evaluation based on the [rubrics_keyword]s and <FULL TEXT> provided, on a scale of 1 to 10. 
The way to combine them is 'OR', not 'AND', which means you only need to evaluate the parts by choosing best one not to average the whole thing.
Put '**table**' at the start of the output
Remember NEVER use apostrophe in the output message. 
Exactly follow the output form and fill in the (score) on a scale of 1 to 10.

<fewshot>

**table**

[["rubrics_keyword_1", "rubrics_keyword_2", "rubrics_keyword_3", "rubrics_keyword_4", "rubrics_keyword_5"], [6, 10, 6, 5, 4]]

<output form>

**table**

[[{rubrics_keyword}], [(rubrics_keyword_1 score of the walking), (rubrics_keyword_2 score of the walking), (rubrics_keyword_3 score of the walking), ...]]



### Task 2
Make the overall opinion (summary of the evaluation) and total score on a scale of 1 to 10, based on the <FULL TEXT> provided. 
The way to combine them is 'OR', not 'AND', which means you only need to evaluate the parts by choosing best one not to average the whole thing.
Remember NEVER use apostrophe in the output message. 


<output form>

**Total score** : 1~10 / 10

**Overall opinion**
(Explain how that above 'Total score' was calculated)

----END of the summary----




"""