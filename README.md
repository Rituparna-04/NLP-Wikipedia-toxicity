# NLP-Wikipedia-toxicity

Problem Statement: 

Wikipedia is the world’s largest and most popular reference work on the internet with about 500 million unique visitors per month. It also has millions of contributors who can make edits to pages. The Talk edit pages, the key community interaction forum where the contributing community interacts or discusses or debates about the changes pertaining to a particular topic. 
Wikipedia continuously strives to help online discussion become more productive and respectful. You are a data scientist at Wikipedia who will help Wikipedia to build a predictive model that identifies toxic comments in the discussion and marks them for cleanup by using NLP and machine learning. Post that, help identify the top terms from the toxic comments. 

Domain: Internet

Analysis to be done: Build a text classification model using NLP and machine learning that detects toxic comments.

Content: 
id: identifier number of the comment
comment_text: the text in the comment
toxic: 0 (non-toxic) /1 (toxic)

Steps to perform:
Cleanup the text data, using TF-IDF convert to vector space representation, use Support Vector Machines to detect toxic comments. Finally, get the list of top 15 toxic terms from the comments identified by the model.
