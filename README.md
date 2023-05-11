# TubeTrend-Pro-Predicting-YouTube-Video-Popularity

**1.1 General Introduction**

As one of the most widely used video-sharing platforms, YouTube has
emerged as a brand-new career path in recent years. YouTubers can make
money through selling items, receiving sponsorships from businesses,
running advertisements on their videos, and receiving donations from
their fans. The success of videos becomes the top concern for YouTubers
in order to keep a steady income. Some of our friends, meanwhile, run
YouTube channels or run other video-sharing websites. This increases our
curiosity about projecting the video\'s performance. In order to get the
greatest attention from the public, artists may modify their videos if
they can make an early estimate about how well they would perform. The
number of people who view a YouTuber\'s video the most worries them. As
a result, based on the quantity of views, the videos can be divided into
popular and unpopular categories.

Feedback from viewers is also crucial for YouTubers because it shows the
preferences of views.

**1.2 Problem Statement**

The objective of this project is to develop a machine learning model
capable of predicting the expected number of views for a forthcoming
YouTube video that a content creator plans to publish in the near
future. This will be a significant challenge as it requires analyzing
multiple factors that contribute to a video\'s popularity, such as
title, thumbnail and past performances of other videos of the same
channel.

**1.3 Significance of the Problem**

Since there is a lot of data available today, we can use a variety of
machine learning algorithms to search for hidden patterns. A video\'s
potential popularity can be determined using the hidden patterns. The
youtubers will directly benefit from this since they will know which
videos will produce the best results for them and which kinds of content
are most popular with the current generation.

1

**1.4 Brief Description of the Solution Approach**\
For making this predictive model we are going to train 3 different
machine learning models Since a viewer\'s decision to watch a video
depends largely on the title and the thumbnail of the video therefore we
are going to train two different models to identify the same category
wise.

So, the First model will be trained on to find whether the video title
is clickbait or not.

The Second model will be trained to identify whether a thumbnail is an
attractive thumbnail or not.

And lastly, we'll use the Youtuber's past videos performances to predict
the number of views on the video that he is going to upload in the near
future.

2

**CHAPTER-2**

**LITERATURE SURVEY**

**2.1 Summary of papers studied**

**Machine Learning enabled models for YouTube Ranking Mechanism and
Views**\
**Prediction\[1\]**\
The paper specializes in the problem of predicting the ranking mechanism
and perspectives of videos on YouTube, that\'s crucial for content
material creators and platform proprietors. The authors suggest a hybrid
machine getting to know model that mixes each content material-based
totally and collaborative filtering techniques to improve the accuracy
of predictions.

The content material-based technique considers the functions of the
video itself, consisting of identify, description, and tags, while the
collaborative filtering technique considers the user\'s conduct and
alternatives, which includes their viewing history and scores. The
proposed hybrid model combines each technique to conquer their barriers
and attain better accuracy in predictions.

To compare the overall performance of their model, the authors
accumulated facts from YouTube and used it to train and test their
model. The results confirmed that the proposed version outperformed
other current fashions and carried out high accuracy in predicting the
ranking mechanism and views of motion pictures on YouTube.

Overall, the examination demonstrates the capacity of device learning in
enhancing the overall performance of YouTube\'s ranking mechanism and
predicting the views of films. This has important implications for
content creators, as it can help them optimize their movies for higher
visibility and engagement, and for platform proprietors, as it is able
to improve the personal experience and growth consumer retention.

3

**Predicting the popularity of online content with knowledge-enhanced
neural networks \[2\]**

In this research paper, the authors proposed a knowledge-enhanced neural
network model that leverages external information sources to improve the
accuracy of predictions for analyzing online content. They used two
types of external information sources, structured knowledge graphs and
unstructured textual content data, to provide additional context and
information about the content being analyzed.

The structured knowledge graphs helped in formalizing the relationships
between entities and concepts in the content, and the authors used this
information to enrich the content representation with entity types,
categories, and attributes. On the other hand, the unstructured textual
content data captured the semantic meaning and context of the content,
and the authors used a technique called phrase embeddings to map each
word in the content to a high-dimensional vector that captures its
semantic meaning.

The authors evaluated the performance of their proposed model on various
datasets from different online platforms and found that it outperformed
other modern models in predicting the popularity of online content. This
suggests that incorporating external information sources into deep
neural networks can improve the accuracy of predictions and enhance the
overall performance of content recommendation systems.

Overall, this study highlights the potential of knowledge-enhanced
neural networks in predicting the popularity of online content and
providing valuable insights for content creators, publishers, and
marketers.

4

**A multimodal variational encoder-decoder framework for micro-video
popularity prediction \[3\]**

The research paper proposes a multimodal variational encoder-decoder
framework for predicting the recognition of micro-videos. Micro-movies
are short videos that typically closing much less than one minute and
are famous on social media systems together with TikTok, Instagram, and
Vine.

The proposed framework combines each visual and textual features of the
micro-movies to teach the version. The visible features include color
histograms, texture capabilities, and motion functions, at the same time
as the textual features encompass captions, hashtags, and person
feedback. The authors use a variational autoencoder (VAE) to study the
joint distribution of the functions and then use a decoder to expect the
recognition of the micro-movies.

The VAE is a type of generative version that learns to generate new
records samples that resemble the education statistics distribution. The
VAE includes an encoder and a decoder network, in which the encoder maps
the enter features right into a low-dimensional latent space, and the
decoder maps the latent area lower back to the enter features. The
authors use the VAE to model the joint distribution of the visible and
textual functions, which lets in them to capture the correlations and
interactions among the distinctive functions.

The authors evaluated the performance of the proposed framework on a
big-scale dataset of micro-films. They compared their version to
numerous different modern day models and showed that the proposed model
outperformed all other models in predicting the popularity of
micro-videos. The take a look at highlights the capacity of multimodal
gaining knowledge of in enhancing the accuracy of reputation prediction
models for micro-motion pictures, and it gives a useful framework for
content material creators and marketers to expect the success of their
micro-video campaigns.

5

**Multi-modal Variational Auto-Encoder Model for Micro-video Popularity
Prediction \[4\]**

The studies paper proposes a machine studying version that predicts the
popularity of micro-videos and the usage of a multi-modal variational
vehicle-encoder (MVAE) method. Micro-films are short motion pictures
that are typically much less than a minute in duration and are commonly
determined on social media structures such as Instagram, TikTok, and
YouTube.

The proposed MVAE model is designed to capture the various factors that
make a contribution to the recognition of micro-movies, including visual
and textual features. The model consists of two components: an encoder
network and a decoder community. The encoder community takes within the
enter features (i.E., visible and textual features) and maps them to a
decrease-dimensional latent area. The decoder community then generates
predictions of the micro-video recognition from the found out latent
area.

The MVAE version is educated on the usage of a variational
vehicle-encoder framework, which allows the model to generate new
predictions with the aid of sampling from the discovered latent area.
This method enables the MVAE version to seize the interactions between
exceptional modalities of enter features and generate correct
predictions of micro-video recognition.

One of the key capabilities of the MVAE version is its ability to deal
with more than one modalities of enter functions. The model makes use of
a joint embedding community that learns a shared representation of the
visual and textual features, which could seize the interactions among
specific modalities.

The paper provides an experimental assessment of the proposed MVAE
version on a large-scale micro-video dataset. The take a look at
suggests that the MVAE version outperforms numerous today\'s methods for
micro-video popularity prediction and affords insights into the
significance of various modalities in predicting micro-video popularity.

Overall, they have a look at and propose a unique method to micro-video
reputation prediction that mixes visible and textual capabilities in a
multi-modal variational car-encoder model. The model is designed to
seize the different factors that contribute to micro-video reputation
and provides a greater correct prediction of reputation as compared to
current strategies.

6

**Can social features help learning to rank youtube videos? \[5\]**

The paper "Can Social Features Help Learning to Rank YouTube Videos?"
explores the capacity of the usage of social functions to improve the
accuracy of rating algorithms for YouTube motion pictures. These social
features are primarily based on user engagement with the videos,
together with likes, dislikes, perspectives, and feedback.

One essential component of the paper is the evaluation of different
mastering to rank algorithms, particularly pointwise, pairwise, and
listwise approaches. These algorithms range within the way they use the
relevance judgments of the education statistics to optimize the rating
feature. The authors locate that incorporating social functions can
improve the rating overall performance of all 3 algorithms, but the
degree of improvement varies depending on the set of rules.

The experiments additionally display that the effectiveness of social
features depends at the availability and great of text inside the video.
When the video includes confined textual data, social functions can
provide a valuable signal to enhance the ranking overall performance.
For example, the wide variety of views or likes can indicate the
recognition of the video, at the same time as the comments can provide
insights into the user sentiment and engagement.

Moreover, the paper analyzes the impact of various social features at
the rating overall performance. The authors discover that likes and
perspectives are the most informative functions, even as dislikes and
feedback have a weaker effect. This suggests that the positive comments
from users (likes and views) includes more weight in ranking than poor
feedback (dislikes) or personal feedback.

In the end, the paper demonstrates that social features can be a
powerful supply of facts for getting to know to rank algorithms for
YouTube movies. The results advocate that incorporating social functions
can improve the rating overall performance, especially while text is
restrained. The paper additionally highlights the importance of choosing
an appropriate getting to know to rank set of rules and social functions
to optimize the overall performance of the video ranking device.

7

**Youtube Videos Prediction: Will this video be popular \[6\]**

The study focuses on creating a machine learning model that has the
ability to predict the success of YouTube videos before they are
uploaded. As video content continues to rise on platforms such as
YouTube, this can be particularly useful for content creators and
advertisers. The researchers collected a dataset of over 10,000 YouTube
videos, which includes several features such as video length, title,
tags, and descriptions.

To develop the machine learning model, the researchers utilized the
Random Forest algorithm, which is widely used for classification and
regression tasks. The algorithm was trained on the dataset of YouTube
videos, with the aim of determining whether a video would be popular or
not. The study classified a video as popular if it attained over 100,000
views within a week of being uploaded.

The machine learning model demonstrated an accuracy of 84% in predicting
the popularity of YouTube videos, which is a significant improvement
compared to traditional methods that rely on simple statistical models
or heuristics. The study also identified key features that were crucial
in predicting the success of a video. These features included the number
of views and likes, the video\'s duration, and the presence of specific
keywords in the title and description.

The study demonstrates the potential of machine learning techniques in
predicting the popularity of online videos. This could be particularly
helpful for content creators looking to enhance their video titles,
descriptions, and tags to increase their chances of success. Advertisers
can also benefit by targeting their ads to videos that are likely to be
popular among their target audience. Furthermore, the research paper
provides a framework for future studies that aim to enhance the accuracy
of video popularity prediction models.

8

**Pay Attention to Virality: understanding popularity of social media
videos with the attention mechanism \[7\]**

The research paper titled \"Pay Attention to Virality: Understanding
Popularity of Social Media Videos with the Attention Mechanism\" by Adam
Bielski and Tomasz Trzcinski examines the use of attention mechanisms
for predicting the popularity of social media videos. The study utilized
a dataset of videos from YouTube that includes different characteristics
such as video length, title, tags, and descriptions.

The authors created a machine learning model that leverages an attention
mechanism to focus on the most significant features of the video to
predict its popularity. This attention mechanism helps the model to
learn which features are most relevant to the popularity of the video
and assign them greater weight in the prediction.

The study discovered that the attention mechanism notably enhances the
accuracy of the model in predicting the popularity of social media
videos. The essential features in predicting video popularity were the
number of views, likes, comments, video length, and title.

Overall, the research paper demonstrates the potential of attention
mechanisms in predicting the popularity of social media videos. This
finding could be useful for content creators who want to optimize their
video titles, descriptions, and tags to increase their chances of
success. The paper provides a framework for future studies that aim to
improve the accuracy of video popularity prediction models using
attention mechanisms.

9

**Recurrent neural networks for online video popularity prediction
\[8\]**

Recurrent Neural Networks (RNNs) are a sort of neural community that can
successfully manner sequential data through retaining a hidden state
that captures previous inputs. This hidden nation can be updated at
every time step based totally at the modern-day center and the previous
hidden nation. The authors of the paper endorse the use of RNNs for
predicting the recognition of on-line videos.

The proposed technique entails the use of a stacked RNN structure with
LSTM (Long Short-Term Memory) cells, that are a sort of RNN cellular
that could capture both short-term and lengthy-time period temporal
dependencies within the facts. The LSTM cells can consider facts over
longer time durations by using selectively updating and forgetting
information inside the hidden state.

The authors explore one-of-a-kind enter representations for the RNN
version, which include uncooked video frames and pre-extracted features.
They evaluate the overall performance of the version on a massive
dataset of on line videos, which incorporates various features together
with audio, visible, and textual facts.

The effects of the experiments display that the proposed RNN method
outperforms other baseline methods for predicting video reputation. The
authors also carry out an ablation to investigate the significance of
different components of the version, consisting of the quantity of
stacked layers and the type of enter representation.

Overall, the paper demonstrates the effectiveness of using RNNs with
LSTM cells for predicting online video reputation, and highlights the
significance of considering each brief-time period and long-time period
temporal dependencies in the records.

10

**Clickbait Detection for YouTube Videos \[9\]**

The paper addresses the issue of clickbait on YouTube, which could lead
to misleading or inappropriate content being advocated to customers, and
might negatively effect the user experience. The proposed answer is a
gadget getting to know-based approach for detecting clickbait in video
titles, which can be used by YouTube or other social media platforms to
routinely filter clickbait content.

To broaden and evaluate their method, the authors use a dataset of video
titles, accrued from YouTube channels and labeled as both clickbait or
no longer clickbait. They extract diverse features from the titles, such
as the frequency of certain phrases or phrases, and use these features
to educate a classification model which could predict whether a given
title is clickbait or not.

The authors experiment with exceptional feature extraction strategies
and classification algorithms, and evaluate their method using diverse
metrics, including accuracy, precision, and do not forget. The
consequences display that their proposed method achieves high accuracy
in detecting clickbait, outperforming different methods proposed in
preceding research.

The authors also conduct an evaluation of the characteristics of
clickbait titles, consisting of the usage of certain keywords and
phrases, and provide insights that would be utilized by YouTube or other
social media systems to enhance their content material recommendation
algorithms. For instance, they advise that clickbait titles often use
emotional or sensational language, and that YouTube could consider
reducing the load given to those styles of titles when recommending
content material to customers.

Overall, the paper provides a beneficial contribution to the field of
clickbait detection and could have realistic implications for improving
the personal experience on social media systems.

11

**Detection of Clickbait Thumbnails on YouTube Using Tesseract-OCR, Face
Recognition, and Text Alteration \[10\]**

The presented research paper proposes a method for detecting clickbait
thumbnails on YouTube using various techniques such as Tesseract-OCR,
face recognition, and text alteration. Clickbait refers to the use of
misleading or sensational content in images or videos to attract clicks
and views, often leading to irrelevant or low-quality content.

The authors collected a dataset of YouTube thumbnails labeled as either
clickbait or not clickbait and used it to train and evaluate their
approach. They employed Tesseract-OCR to extract text from the
thumbnails and face recognition techniques to identify whether the
thumbnail contains faces or not. They also used text alteration
techniques to detect whether the text in the thumbnail has been altered
or distorted.

The authors experimented with various feature extraction techniques and
machine learning algorithms and evaluated their approach using various
metrics such as accuracy, precision, and recall. The results
demonstrated that their proposed method achieved high accuracy in
detecting clickbait thumbnails, outperforming other methods proposed in
previous studies.

Furthermore, the authors discuss the potential applications of their
approach, such as helping YouTube or other social media platforms to
automatically filter out clickbait content and improve the user
experience. They also suggest that their approach could be extended to
detect other types of misleading or harmful content, such as fake news
or hate speech.

The authors utilized the SVM Model to process the implementation
results. Their dataset comprised 250 thumbnails, resulting in an
accuracy value of 0.968, a sensitivity value of 0.968, a precision value
of 0.9698, and an F1-Score of 0.9678.

Overall, the paper provides a significant contribution to the field of
clickbait detection and could have practical implications for improving
the quality of content on social media platforms.

12

**Clickbait detection using deep learning \[11\]**

The paper proposes a solution to detect clickbait headlines using Deep
Learning techniques. Clickbait headlines often use misleading or
sensationalist content to lure readers into clicking on an article,
which can harm the credibility of news sources and provide low-value
content. To address this issue, the paper suggests using a Convolutional
Neural Network (CNN) to extract features from the headline text. CNNs
are effective at learning patterns in images and sequences, making them
suitable for learning patterns in text. The output of the CNN is then
fed into a fully connected neural network for classification, which
predicts whether the headline is clickbait or not.

The authors also explore the use of word embeddings and attention
mechanisms to enhance the performance of the model. Word embeddings
represent words as vectors in a high-dimensional space, which captures
the meaning and context of the words. Attention mechanisms enable the
model to focus on different parts of the headline text, based on their
relevance to the classification task. The approach is evaluated on a
large dataset of clickbait and non-clickbait headlines, and the results
indicate that the proposed model can effectively detect clickbait with
high accuracy. Additionally, an ablation study is conducted to
investigate the importance of different components of the model, such as
the use of word embeddings and attention mechanisms.

Overall, the paper demonstrates the potential of Deep Learning
techniques, particularly CNNs, for clickbait detection. By identifying
clickbait headlines, the proposed approach can improve the credibility
and quality of online content.

13

**The good, the bad and the bait: Detecting and characterizing clickbait
on youtube \[12\]**

The research paper addresses the issue of detecting and characterizing
clickbait on YouTube. Clickbait refers to sensational or misleading
content aimed at attracting clicks and views, which often results in
low-quality or irrelevant content. The authors collect a vast dataset of
YouTube videos and use a combination of manual labeling and machine
learning-based techniques to identify clickbait content. They also
examine the characteristics of clickbait videos, such as the use of
specific keywords or phrases, and compare them to non-clickbait videos.

The findings suggest that clickbait videos are more likely to use
specific words or phrases, such as \"you won\'t believe\" or
\"shocking,\" and that they often have certain attributes, such as
shorter video length and higher view count. Moreover, clickbait videos
are more likely to contain misleading or irrelevant content. The authors
discuss the implications of their findings for YouTube and other social
media platforms, highlighting that their approach could help filter out
clickbait content and enhance the user experience. They also note that
their findings could be used to improve content recommendation
algorithms, thus minimizing the spread of clickbait content.

Overall, the paper presents a valuable contribution to the field of
clickbait detection and characterization on YouTube, with practical
implications for improving the quality of content on social media
platforms.

14

**Baitradar: A multi-model clickbait detection algorithm using deep
learning \[13\]**\
The research paper proposes an approach named BAITRADAR for clickbait
detection, which is a multi-model deep learning approach that combines
Convolutional Neural Networks (CNNs) and Recurrent Neural Networks
(RNNs). The CNNs extract local features by applying filters to identify
patterns within short sequences of words, while the RNNs capture the
global context of the headline by processing the text sequentially and
maintaining a hidden state that captures the previous words in the
headline. By combining both CNNs and RNNs, the proposed approach
captures both local and global features of the headline text, which is
crucial for clickbait detection.

The authors also use a stacked ensemble model to further improve the
performance of the individual models. The stacked ensemble model
combines the outputs of the individual models and uses another neural
network to learn how to weigh the individual model predictions. This
approach has been shown to be effective in improving the performance of
machine learning models.

The experiments are conducted on a large dataset of clickbait and
non-clickbait headlines, and the results demonstrate that the proposed
BAITRADAR approach achieves high accuracy in detecting clickbait.
Additionally, the authors conduct an ablation study to investigate the
importance of different components of the model, such as the number of
layers in the CNN and RNN models. Overall, the paper highlights the
effectiveness of a multi-model deep learning approach for clickbait
detection. The proposed BAITRADAR approach can capture both local and
global features of the headline text, and it achieves high accuracy in
detecting clickbait, which can be beneficial in enhancing the quality of
online content.

15

**Thumbnail Image Design to Grab Views in Online Video Platform:
Evidence from YouTube \[14\]**

The studies paper investigates the layout of thumbnail photos in on-line
video systems, mainly on YouTube. The look examines the connection
between thumbnail image characteristics and video perspectives. The
authors collected statistics on over 1,three hundred YouTube movies and
analyzed the thumbnail pix and their attributes inclusive of shade,
text, and image content. The outcomes display that videos with more
shiny colorations, excessive comparison, and extra text tend to have
better views. Additionally, motion pictures with photographs of human
faces or close-up pictures of gadgets additionally tend to have higher
views. They conclude that thumbnail photos play an important function in
attracting viewers and growing video perspectives. The findings of these
studies may be beneficial for content creators and marketers in
designing effective thumbnail pix to book their online visibility and
engagement.

16

**Classifying YouTube Videos by Thumbnail \[15\]**

In this research paper titled \"Analyzing the Impact of Thumbnail Images
on YouTube Video Classification,\" the authors investigate the use of
thumbnail images as a means of categorizing videos on YouTube. They
propose a deep learning model that analyzes thumbnail images and assigns
videos to categories such as news, sports, music, and gaming.

To test their model, the authors utilized a dataset of over 450,000
YouTube videos, each with its corresponding thumbnail image. They
trained and tested their model on this dataset and achieved a high
accuracy rate of over 90% in correctly classifying videos into different
categories based solely on their thumbnail images.

The authors also conducted an analysis of the most common features found
in thumbnail images for each video category. For instance, they observed
that music video thumbnails tended to have bright colors and feature
images of the artist or band, while news video thumbnails were more
likely to feature people and text.

The study concludes that thumbnail images can serve as a useful tool for
categorizing videos on YouTube, and that deep learning techniques can be
effective in analyzing and extracting features from these images. These
findings can be valuable for content creators and marketers seeking to
design effective thumbnail images that enhance the online visibility and
engagement of their videos.

17

**The Impact of YouTube\'s Thumbnail Images and View Counts on Users\'
Selection of Video Clip, Memory Recall, and Sharing Intentions of
Thumbnail Images \[16\]**

\"The Impact of YouTube\'s Thumbnail Images and View Counts on Users\'
Selection of Video Clip, Memory Recall, and Sharing Intentions of
Thumbnail Images\" is a research paper that investigates how thumbnail
images and view counts influence users\' selection of video clips,
memory recall, and sharing intentions on YouTube. The study was
conducted using an online survey with 400 participants, and various
combinations of thumbnail images and view counts were presented to the
participants. They were then asked to choose which video they would
watch, rate their memory recall of the video content, and state their
intentions to share the video.

The results of the study suggest that users are more likely to choose a
video with a visually appealing, emotionally evocative, and relevant
thumbnail image. Additionally, high view counts are associated with
greater perceived popularity and credibility of a video, leading to
increased intentions to watch and share the video. Furthermore, videos
with visually appealing and emotionally evocative thumbnail images are
better recalled by users, and users are more likely to share videos with
relevant and visually appealing thumbnail images.

Overall, the research highlights the significance of thumbnail images
and view counts in shaping user behavior on YouTube. The findings can be
beneficial for content creators and marketers in creating effective
thumbnail images and enhancing their online visibility and engagement.

18

**Visual Attributes of Thumbnails in Predicting Top YouTube Brand
Channels: A Machine Learning Approach \[17\]**

The research paper, \"Visual Attributes of Thumbnails in Predicting Top
YouTube Brand Channels: A Machine Learning Approach.\" explores how
visual attributes of thumbnail images can be used to predict the success
of YouTube brand channels. The study analyzed a dataset of more than
8,000 thumbnail images from the top 500 YouTube brand channels and
employed deep learning techniques to identify visual attributes
associated with success. The authors then used this dataset to train and
test their machine learning model to predict successful brand channels
based on their thumbnail images.

The research revealed that certain visual attributes of thumbnail images
were more likely to be associated with successful YouTube brand
channels. These attributes included brighter and more vivid colors,
larger text, and a clear focal point. Additionally, thumbnail images
featuring human faces were found to be more successful than those
without. The machine learning model developed in the study achieved an
accuracy of over 80% in accurately predicting the success of YouTube
brand channels based solely on their thumbnail images. This research can
be useful for content creators and marketers in designing effective
thumbnail images and increasing the success of their brand channels on
YouTube.

19

**Analysis of YouTube thumbnails: a deep neural decision forest
implementation \[18\]**\
In the research paper titled \"Analysis of YouTube Thumbnails: A Deep
Neural Decision Forest Implementation,\" the authors investigate the
application of deep neural decision forests (DNDF) to analyze and
categorize YouTube thumbnail images. To conduct the study, the authors
utilized a dataset of over 10,000 YouTube videos and their corresponding
thumbnail images, and trained and tested their model using a DNDF, which
combines decision tree and neural network techniques. Through this
approach, the study was able to accurately categorize thumbnail images
into various categories such as music, sports, and news, achieving an
accuracy rate of over 90%. Additionally, the authors analyzed the most
common visual features of thumbnail images for each video category.
Overall, the study highlights the potential of using DNDF for effective
analysis and classification of YouTube thumbnail images, which can be
valuable for content creators and marketers seeking to enhance their
online visibility and engagement.

20

**What makes an image popular? \[19\]**

The research paper titled \"What makes an image popular?\" investigates
the factors that contribute to an image\'s popularity on social media
platforms like Flickr and Twitter. To achieve this, the study analyzed
over 2.3 million images and their associated metadata, using machine
learning techniques to model the relationship between image features and
popularity.

The study found that a combination of visual and non-visual factors
influence the popularity of images. Visual factors such as colorfulness,
brightness, and sharpness were found to have a positive correlation with
popularity. Additionally, images that feature faces, interesting
textures, and patterns were more likely to be popular. Non-visual
factors such as the time of day and day of the week an image was posted,
as well as the number of comments and tags, were also significant
predictors of image popularity.

The study also discovered that different types of images tend to be
popular on different social media platforms. For instance, images
featuring people tend to be more popular on platforms like Flickr,
whereas images of food tend to perform better on platforms like
Instagram.

Overall, the research emphasizes the importance of considering both
visual and non-visual factors when creating content for social media.
The findings of this study can be valuable for content creators and
marketers in designing effective visual content that is more likely to
be popular on social media platforms.

21

**2.2 Integrated summary of the literature studied**

**Gaps in research paper:**\
1)Some of the research paper did not consider one of the most important
factor i.e. title and thumbnail which decides whether user is going to
watch the video or not\
2)Several research papers have attempted to predict the view count of
videos based on metrics such as likes count, dislike count, and comment
count. However, it should be noted that these metrics are not available
before the video is published. Therefore, attempting to predict the view
count before the video\'s release using these metrics is not a practical
approach.

**Summary**:\
The thumbnail is the first thing that viewers see when they come across
a video on YouTube, and it can have a significant impact on whether they
click on the video or not. A visually appealing and attention-grabbing
thumbnail can increase the likelihood of a viewer clicking on the video
and watching it. Creators should aim to create thumbnails that
accurately represent the content of the video while also being visually
appealing and attention-grabbing.

The title of a YouTube video is also crucial in determining its
popularity. A clear and descriptive title that accurately reflects the
content of the video can make it easier for viewers to find and watch
the video. Additionally, creators can use clickbait titles to generate
curiosity and intrigue among viewers, which can increase the likelihood
of the video being clicked on and watched. However, it\'s important to
ensure that the title accurately reflects the content of the video to
avoid disappointing viewers and damaging the creator\'s reputation.

The number of views is perhaps the most fundamental metric used to
measure the popularity of a YouTube video. The more views a video has,
the more popular it is considered to be. High view counts indicate that
the video is engaging, informative, or entertaining and that it has
resonated with a large audience.

The number of likes is another important metric used to measure the
popularity of a YouTube video. Likes indicate that viewers have enjoyed
the video and found it valuable, entertaining, or informative. A higher
number of likes can lead to higher rankings in YouTube\'s search results
and recommended videos, which can increase the visibility and popularity
of the video.

22

**CHAPTER-3**

**REQUIREMENT ANALYSIS & SOLUTION APPROACH**

**3.1 Overall description of the project**

This project consists of three parts i.e **Clickbait Thumbnail
Detection, Clickbait Title Detection and Prediction of views using past
videos performances.**

In the first part, there was no publicly available dataset to train our
model. So we've to create a dataset on our own having thumbnails of
videos uploaded by 70-80 Youtubers.

Then we trained our model to predict whether the thumbnail uploaded by
the user is clickbait or not.

In the second part, we've a dataset of about 32000 titles , which we
used to train our model using different machine learning algorithms and
select the one which gives the best accuracy.

In the third part,This model utilizes both the recent performance of the
last 10 videos, which is measured by their views, engagement rate, and
duration, as well as the metadata related to the YouTube channel such as
subscriber count, number of videos published, duration of the current
video, and the clickbait rating of both the title and thumbnail (which
is calculated using previous models). The objective of this model is to
predict the view count for the current video. Therefore, the third model
is a regression model that uses the above-mentioned metadata to estimate
the view count of the video.

**3.2 Requirement Analysis**\
**3.2.1 Software Requirements**\
●Operating System: Linux x64 (Ubuntu 20.04 and later) or Windows 8.1 and
later ●Language: Python 3.7\
●Libraries Used:\
○Scikit-learn\
○sklearn\
○Matplotlib\
○MarkupSafe\
○Numpy

23

> **3.2.2 Hardware Requirements**
>
> ● CPU: 2GHz processor (minimum)\
> ● Computer Processor: Intel i5 or i7 core/ Ryzen 5 or 7
>
> ● Computer Memory:
>
> ○ RAM: 2GB or more
>
> ○ HDD: 10GB or more\
> ● Graphics Hardware: Not mandatory

**3.3 Solution Approach**

The project consists of three models.

In order to account for the significant influence of video title and
thumbnail in determining user interest, the project has placed a special
emphasis on analyzing these two metadata features. As a result, two
separate models have been developed to predict the effectiveness of a
given title and thumbnail.

The third model is a regression model which takes into account the view
count of past 10 videos as

well as the engagement gained on these videos. We'll also be using the
above two models to predict the view count.

**First Model:**

To find out whether a video is clickbait or not, we have to use a
dataset containing two columns, the first one is title of the video and
the second column will tell us whether the title is clickbait or not(0
-\> not clickbait, 1-\> clickbait). To train the model for classifying
clickbait titles, first we need to preprocess the dataset.

Preprocessing includes these major steps:

> 1)Tokenization of Data\
> 2)Removing extra spaces\
> 3)Removing punctuations\
> 4)Removing stopwords (These words have little to no significance in
> determining the meaning or sentiment of a text and are often highly
> frequent in occurrence)

24

After preprocessing the data, we will perform feature extraction and
train the model using various machine learning algorithms such as Naive
Bayes, Linear regression, Decision tree, etc.

**Linear Regression:**\
Linear regression is a simple and commonly used statistical method to
model the relationship between a dependent variable and one or more
independent variables. The goal of linear regression is to find the best
linear equation that explains the relationship between the variables.
The equation can be used to make predictions about the dependent
variable for new values of the independent variables. Linear regression
assumes that the relationship between the variables is linear and that
the errors are normally distributed.

**SVM Regression:**\
Support Vector Machine (SVM) Regression is a regression technique that
uses a subset of training points (support vectors) to create a
regression model. The goal of SVM regression is to find the best
hyperplane that fits the training data while maximizing the margin
between the hyperplane and the training points. SVM regression can
handle non-linear relationships by using a kernel function to transform
the data into a higher-dimensional space where it can be linearly
separable.

**Decision Tree Regression:**\
Decision Tree Regression is a non-parametric regression technique that
uses a tree-like model to predict the value of a dependent variable
based on the values of several independent variables. Decision trees are
constructed by recursively splitting the data into smaller subsets based
on the value of an independent variable that minimizes the variance of
the dependent variable. Decision tree regression can handle non-linear
relationships and can easily handle categorical variables.

**XGBoost Regression:**\
XGBoost Regression is a popular implementation of gradient boosting that
uses decision trees as base learners. Gradient boosting is a technique
that combines weak learners (simple models that perform slightly better
than random guessing) to create a strong learner. XGBoost uses a
weighted sum of decision trees to predict the value of a dependent
variable. XGBoost is known for its speed and accuracy and is commonly
used in machine learning competitions.

The algorithm which will give best accuracy, recall and f1-score will be
finally selected.

25

**Second Model:**\
From the research papers studied there are few important criterias for
good thumbnail:\
i) High Contrast Images\
ii) Vivid and Bright colors used in Images\
iii) Contains Human Faces\
iv) Close Up Shot Of An Object\
v) Contains Text Material\
We've selected some 70-80 channels which uploads videos having clickbait
thumbnails and extracted the thumbnails using a python script.

Similarly we've selected some 70-80 channels which uploads videos having
normal thumbnails (not clickbait).

This way we've created a dataset having two classes: clickbait and
not-clickbait.

And at the end we will train the model using the obtained features
against the target variable i.e good thumbnail or bad thumbnail.

We will use CNN for model training.

CNN stands for Convolutional Neural Network, which is a type of
artificial neural network used primarily for analyzing visual imagery.
It is a deep learning algorithm designed to recognize and classify
images based on the patterns and features present in the images.

The key feature of CNNs is their ability to perform convolution
operations, which are mathematical operations that extract specific
features from an image. These features are then combined to classify the
image. CNNs also use pooling layers to reduce the size of the data and
make the computation more efficient.

CNNs are composed of several layers, including input, convolutional,
pooling, fully connected, and output layers. In the input layer, the
image data is fed into the network. In the convolutional layer, the
network applies a set of filters to the image to extract features. The
pooling layer reduces the dimensionality of the feature map. The fully
connected layer combines the features to generate a prediction, and the
output layer provides the final classification result.

26

> CNNs are widely used in a variety of applications, such as image and
> speech recognition, natural language processing, and self-driving
> cars. They have shown to be highly effective in many tasks, including
> object detection, face recognition, and image segmentation, among
> others.

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image2.png){width="6.583332239720035in"
height="2.4694444444444446in"}

1\. Convolution Neural Network

> **Third model:**\
> In this model we are taking the past performance of last 10 videos (
> performance measured from video views , engagement rate and duration)
> , also we are taking the metadata related to channel i.e.
>
> Subscriber count , number of videos published by the youtube channel ,
> duration of the current video and title clickbait ness as well as the
> thumbnail clickbait ness (whose values are calculated using the last
> two models ) . Here our target variable is the view count of the
> current video.
>
> So our third model is a regression model that is going to predict the
> view count using the above metadata.

27

**CHAPTER-4**

> **MODELING AND IMPLEMENTATION DETAILS**

**4.1 Design Diagrams**

> **4.1.1 Control Flow Diagrams**
>
> ![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image3.png){width="5.084722222222222in"
> height="4.759722222222222in"}

2\. Control Flow Diagram

**Detailed Description of metadata:**

> 1)Channel_id : ID of every youtube channel.
>
> 2)view1, view2 .... view10 : views of previous 10 videos\
> 3)engage1, engage2 ... engage10 : engagement rate of every video which
> is calculated using the formula (likes+comments)/views\
> 4)dur1, dur2 ... dur10 : duration of previous 10 videos\
> 5)no_of_videos : Number of videos uploaded on the channel\
> 6)subsCount : Total number of subscribers on tbe channel

28

> 7)title : title of the video\
> 8)thumbnail: thumbnail of the video\
> 9)ClickbaitThumbnail :binary attribute 0 or 1 whether the thumbnail is
> clickbait or not. 10)ClickbaitTitle :binary attribute 0 or 1 whether
> the title is clickbait or not.

**4.2 Implementation Details and issues**

For the first model, we've to start off with creating a dataset that
contains images having two classes:

> 1)clickbait\
> 2)not-clickbait

So, we've used a list of channels having clickbait and not-clickbait
thumbnails.

Using this list we can use the Youtube API v3 to extract the recent 20
videos of these channels and we can use i1.ytimg.com to download the
thumbnails of these videos.

After downloading the thumbnails, now we've to do dataset division into
test, train and val.

We will be keeping the ratio as 0.7, 0.2 and 0.1 for train, test and val
respectively.

Now the dataset part is complete , we can perform Image classification
to predict the clickbaitness of the thumbnail.

Finally we can save the model to use it further.

For the second model, we've a dataset having dimensions as 32199 rows ×
2 columns

title gives us the title of Youtube video, and isClickbait is a binary
label which denotes whether the title is clickbait or not.

Now, we will use this dataset to train our model. We will use 80% for
training and the remaining 20% for testing purposes.

But before training the data needs to be cleaned.

For data cleaning and preprocessing we will be doing these things:

> 1)Tokenization\
> 2)Converting to lowercase\
> 3)Removing spaces and numbers

29

> 4)Removing stopwords and punctuations\
> 5)Lemmatization

To get the best possible accuracy, we will be using following machine
learning algorithms:

> 1)Naive Bayes\
> 2)Random Forest\
> 3)Decision Tree\
> 4)Logistic Regression\
> 5)XGBoost

And we will pick the algorithm that will give best possible accuracy and
f1-score.

Now we have the model, we can predict the clickbaitness of any title
input by the user.

For the third model:

We've created the dataset by using Youtuber's past 10 videos .

For every video, we've taken count of views of these videos, duration of
every video , subscriber count of the channel and etc.

Then we collected the data using youtube api v3.

This dataset contains 100 rows and in each row, there are 10 columns for
views of previous 10 videos,

then we've 10 rows for duration of previous 10 videos, then we've
engagement rate of 10 videos, title and thumbnail or current video,
subscriber count of the channel.

We performed data cleaning and removed the rows which contained empty
cells

First we applied linear regression on the model and got the score of
0.88 i.e. the model explains 88% of all the variations in the data.

Then we applied Decision tree regression and got a score of 0.77

Then we applied SVR got a score of -0.05 and the we applied Xgboost
regression and got a score of 0.73.

30

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image4.png){width="5.668055555555555in"
height="4.398611111111111in"}

> 3\. Value of MSE at different test size for Linear Regression

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image5.png){width="5.768055555555556in"
height="4.416666666666667in"}

> 4\. Value of MSE at different test size for Decision Tree Regression

31

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image6.png){width="5.668055555555555in"
height="4.416666666666667in"}

> 5\. Value of MSE at different test size for SVR Regression

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image7.png){width="5.668055555555555in"
height="4.391666666666667in"}

> 6\. Value of MSE at different test size for XGBoost Regression

32

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image8.png){width="5.968055555555556in"
height="4.25in"}

> 7\. Value of MAE at different test size for Linear Regression

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image9.png){width="6.101388888888889in"
height="4.275in"}

> 8\. Value of MAE at different test size for Decision Tree Regression

33

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image10.png){width="5.968055555555556in"
height="4.266666666666667in"}

> 9\. Value of MAE at different test size for SVR Regression

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image11.png){width="5.968055555555556in"
height="4.283332239720035in"}

> 10\. Value of MAE at different test size for XGBoost Regression

34

+-----------+-----------+-----------+-----------+-----------+-----------+
|           | Test Size |           |           |           |           |
|           | = 0.15    |           |           |           |           |
+===========+===========+===========+===========+===========+===========+
|           | > R2      | > MSE     | > RMSE    | > MAE     | > MAPE    |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > Linear\ | >         | >         | >         | > 7       | >         |
| > R       |  0.893080 |  8.739996 |  9.348795 | 05601.523 | 1108.8115 |
| egression |           | > x 1011  | > x 105   |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| >         | >         | >         | >         | > 6       | >         |
|  Decision |  0.443647 |  4.547819 |  2.132562 | 37898.466 |  225.5354 |
| > Tree    |           | > x 1012  | > x 106   |           |           |
| > R       |           |           |           |           |           |
| egression |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > SVR     | >         | >         | >         | > 8       | >         |
|           | -0.080534 |  8.832650 |  2.971978 | 99030.341 |  266.5747 |
|           |           | > x 1012  | > x 106   |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| >         | >         | >         | >         | > 4       | > 25.8800 |
|  XGBoost\ |  0.723782 |  2.257900 |  1.502631 | 64904.927 |           |
| > R       |           | > x 1012  | > x 106   |           |           |
| egression |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+

Table 1 : Comparing R2, MSE , RMSE , MAE and MAPE of different models
(Test Size=0.15)

+-----------+-----------+-----------+-----------+-----------+-----------+
|           | Test Size |           |           |           |           |
|           | = 0.20    |           |           |           |           |
+===========+===========+===========+===========+===========+===========+
|           | > R2      | > MSE     | > RMSE    | > MAE     | > MAPE    |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > Linear\ | >         | >         | >         | > 7       | >         |
| > R       |  0.882260 |  7.383092 |  8.592492 | 13903.075 | 1165.6153 |
| egression |           | > x 1011  | > x 105   |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| >         | >         | >         | >         | > 3       | >         |
|  Decision |  0.774152 |  1.416222 |  1.190051 | 46217.800 |  184.0030 |
| > Tree    |           | > x 1012  | > x 106   |           |           |
| > R       |           |           |           |           |           |
| egression |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > SVR     | >         | >         | >         | > 6       | >         |
|           | -0.054941 |  6.615207 |  2.572004 | 90930.055 |  224.6964 |
|           |           | > x 1012  | > x 106   |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| >         | >         | >         | >         | > 3       | > 13.5967 |
|  XGBoost\ |  0.730442 |  1.370822 |  1.300121 | 59701.728 |           |
| > R       |           | > x 1012  | > x 106   |           |           |
| egression |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+

Table 2 : Comparing R2, MSE , RMSE , MAE and MAPE of different models
(Test Size=0.20)

+-----------+-----------+-----------+-----------+-----------+-----------+
|           | Test Size |           |           |           |           |
|           | = 0.25    |           |           |           |           |
+===========+===========+===========+===========+===========+===========+
|           | > R2      | > MSE     | > RMSE    | > MAE     | > MAPE    |
+-----------+-----------+-----------+-----------+-----------+-----------+
| > Linear\ | >         | >         | >         | > 9       | >         |
| > R       |  0.634542 |  1.858486 |  1.363263 | 29504.160 |  957.9705 |
| egression |           | > x 1012  | > x 106   |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| Decision  | >         | >         | >         | > 4       | >         |
| Tree      |  0.462853 |  2.731587 |  1.652751 | 06025.720 |  232.9628 |
|           |           | > x       | > x       |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+

35

+-----------+-----------+-----------+-----------+-----------+-----------+
| > R       |           | > 1012    | > 106     |           |           |
| egression |           |           |           |           |           |
+===========+===========+===========+===========+===========+===========+
| > SVR     | >         | >         | >         | > 5       | >         |
|           | -0.038020 |  5.278710 |  2.297544 | 76733.124 |  214.3888 |
|           |           | > x 1012  | > x 106   |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| >         | >         | >         | >         | > 3       | > 38.9263 |
|  XGBoost\ |  0.730438 |  1.370822 |  1.170821 | 22554.006 |           |
| > R       |           | > x 1012  | > x 106   |           |           |
| egression |           |           |           |           |           |
+-----------+-----------+-----------+-----------+-----------+-----------+

Table 3 : Comparing R2, MSE , RMSE , MAE and MAPE of different models
(Test Size=0.25)

**R2:** The amount of variance in the dependent variable that is
explained by the independent variables in a regression model is measured
statistically as R-squared (R2).

**MSE:** The average squared difference between a target variable\'s
predicted and actual values is measured as the mean squared error. It is
calculated by dividing the total number of observations by the sum of
the squared discrepancies between the expected and actual values.

MSE = (1/n) \* ∑(y - ŷ)\^2

where n is the number of observations, y is the actual value of the
target variable, and ŷ is the predicted value of the target variable.

**RMSE:** The average error between the anticipated and actual values is
often measured using the root mean squared error, which is the MSE\'s
square root. It is frequently provided in the same units as the
objective variable and is used to assess how accurately a model
predicts.

RMSE = sqrt((1/n) \* ∑(y - ŷ)\^2)\
where n is the number of observations, y is the actual value of the
target variable, and ŷ is the predicted value of the target variable.

**MAE:** The average absolute difference between a target variable\'s
expected and actual values is measured as the mean absolute error. It is
determined by dividing the total number of observations by the sum of
the absolute discrepancies between the expected and actual values.

MAE = (1/n) \* ∑\|y - ŷ\|

where n is the number of observations, y is the actual value of the
target variable, and ŷ is the predicted value of the target variable\
36

**MAPE:** The average % difference between a target variable\'s
projected and actual values is measured as the mean absolute percentage
error. It is described as the mean, stated as a percentage, of the
absolute percentage disparities between the expected and actual values.

MAPE = (1/n) \* ∑(\|(y - ŷ)/y\| \* 100%)

where n is the number of observations, y is the actual value of the
target variable, and ŷ is the

predicted value of the target variable.

37

**4.3 Risk Analysis and Mitigation**

+-------------+-------------+-------------+-------------+-------------+
| **Risk_ID** | **Class     | > **        | > **Risk    | >           |
|             | ification** | Description | > Area**    |  **Impact** |
|             |             | > of Risk** |             |             |
+=============+=============+=============+=============+=============+
| > Risk_1    | > Design    | > Reading   | >           | Moderate    |
|             |             | > and       | Performance | (M)         |
|             |             | > writing   |             |             |
|             |             | > to the    |             |             |
|             |             | > disc      |             |             |
|             |             | >           |             |             |
|             |             |  frequently |             |             |
|             |             | > takes     |             |             |
|             |             | > time as   |             |             |
|             |             | > data is   |             |             |
|             |             | >           |             |             |
|             |             | transferred |             |             |
|             |             | > from      |             |             |
|             |             | > primary   |             |             |
|             |             | > to        |             |             |
|             |             | > secondary |             |             |
|             |             | > memory.   |             |             |
|             |             | > The HDD   |             |             |
|             |             | > or SSD    |             |             |
|             |             | > being     |             |             |
|             |             | > used disc |             |             |
|             |             | > seek time |             |             |
|             |             | >           |             |             |
|             |             |  determines |             |             |
|             |             | > p         |             |             |
|             |             | erformance. |             |             |
+-------------+-------------+-------------+-------------+-------------+
| > Risk_2    | >           | > Since     | >           | > Low (L)   |
|             | Engineering | > each      | Reliability |             |
|             | >           | > write and |             |             |
|             | Specialties | >           |             |             |
|             |             | calculation |             |             |
|             |             | > is        |             |             |
|             |             | > performed |             |             |
|             |             | > ind       |             |             |
|             |             | ependently, |             |             |
|             |             | > the       |             |             |
|             |             | > affected  |             |             |
|             |             | > files may |             |             |
|             |             | > only be   |             |             |
|             |             | > partially |             |             |
|             |             | > written   |             |             |
|             |             | > in the    |             |             |
|             |             | > event of  |             |             |
|             |             | > an error  |             |             |
|             |             | > or crash, |             |             |
|             |             | > and there |             |             |
|             |             | > will be   |             |             |
|             |             | > no way to |             |             |
|             |             | > roll      |             |             |
|             |             | > back.     |             |             |
+-------------+-------------+-------------+-------------+-------------+
| > Risk_3    | > R         | Python has  | C           | > Low (L)   |
|             | equirements | several     | ompleteness |             |
|             |             | runtime     |             |             |
|             |             |             |             |             |
|             |             | <table>     |             |             |
|             |             | <colgroup>  |             |             |
|             |             | <col        |             |             |
|             |             | style="wid  |             |             |
|             |             | th: 33%" /> |             |             |
|             |             | <col        |             |             |
|             |             | style="wid  |             |             |
|             |             | th: 33%" /> |             |             |
|             |             | <col        |             |             |
|             |             | style="wid  |             |             |
|             |             | th: 33%" /> |             |             |
|             |             | </colgroup> |             |             |
|             |             | <thead>     |             |             |
|             |             | <tr         |             |             |
|             |             | clas        |             |             |
|             |             | s="header"> |             |             |
|             |             | <th>require |             |             |
|             |             | ments,</th> |             |             |
|             |             | <th         |             |             |
|             |             | >which</th> |             |             |
|             |             | <th><       |             |             |
|             |             | blockquote> |             |             |
|             |             | <           |             |             |
|             |             | p>could</p> |             |             |
|             |             | </block     |             |             |
|             |             | quote></th> |             |             |
|             |             | </tr>       |             |             |
|             |             | </thead>    |             |             |
|             |             | <tbody>     |             |             |
|             |             | </tbody>    |             |             |
|             |             | </table>    |             |             |
|             |             |             |             |             |
|             |             | > cause     |             |             |
|             |             | > strange   |             |             |
|             |             | >           |             |             |
|             |             |  behaviour. |             |             |
+-------------+-------------+-------------+-------------+-------------+

2\. Risk Analysis

38

**CHAPTER-5**

**TESTING**

**5.1 Testing Plan**

The program comprises several algorithms which are tested individually
for accuracy. we check for the correctness of the program as a whole and
how it performs.

**5.2 Component decomposition and type of testing required**

**Unit Testing**

Unit tests are designed to make sure that when a transaction is
executed, the world state is changed in the appropriate ways.
Transaction processor functions should contain unit tests covering all
of the business logic, ideally with 100% code coverage. This will
guarantee that your business logic is free of typos and logical
mistakes. From a command line, each module can be executed separately
and tested for accuracy. To check the returned answer and compare it to
the values given to him or her, the tester can pass a variety of values.
The alternative workaround is to create a script, use it to perform all
the tests, and then utilise the output to create a log file that can be
used to check the results.

**System Testing**

Software testing at the level known as \"System Testing\" involves
testing an integrated system as a whole. This test\'s objective is to
assess how well the system complies with the given requirements. The
testing of a finished, fully integrated software product is known as
system testing. White Box Testing, too. Software testing\'s black box
testing subcategory includes system testing.

Various System Testing Methods:

> **• Usability Testing**
>
> Usability testing primarily examines how simple it is for users to use
> an application, how flexible it is to handle controls, and how well a
> system can accomplish its goals.

39

**Quality Assurance**

The purpose of quality assurance, also referred to as QA testing, is to
make sure that a company is offering its clients the finest available
goods or services. The goal of QA is to make processes better so that
customers receive high-quality products. An company must make sure that
its procedures meet the quality requirements established for software
products.

**Functional Test**

Functional testing, commonly referred to as functional completeness
testing, entails looking for any potential gaps in functionality.
Functional testing of key chatbot components is necessary as they
develop into new application domains. Use-case scenarios and associated
business processes, such as the operation of smart contracts, are
evaluated through functional testing.

**5.3 List of all test cases**

> 1.Data collection test cases:\
> ●Ensure that the script to scrape video data from YouTube is working
> as expected ●Test the script with different categories of videos to
> ensure that the data is collected correctly\
> ●Verify that the data is stored in the expected format and is
> accessible for analysis 2.Data preprocessing test cases:\
> ●Check for missing values and decide how to handle them (e.g., fill in
> with mean, median, or zero)\
> ●Transform the data into a format suitable for machine learning
> algorithms\
> ●Check for data imbalances and apply data balancing techniques if
> necessary 3.Model training and testing test cases:\
> ●Split the data into training and testing sets\
> ●Train the model on the training set\
> ●Test the model on the testing set and evaluate its performance\
> ●Use different evaluation metrics such as accuracy, precision, recall,
> F1-score,

40

> **5.4 Error and Exception Handling**\
> While downloading the thumbnails two types of error can come:\
> 1)The channel in the dataset is deleted from Youtube\
> 2)Since we're using Youtube API, in the free version it has a limit of
> 10000 credits per day. If any of the above two things happen, error
> will come and our code will be broken.
>
> To handle these errors, we've used try and except block:

![](vertopal_8c068c6a3dce4498a512cd7c33780b3a/media/image12.png){width="6.687498906386701in"
height="3.6666666666666665in"}

> 11\. Error Handling for Channel Not found error While creating the
> dataset for our third model, we face two types of errors: 1)If the
> values are not available from the API, then we fill it by np.nan()
> 2)If the limit of API reached then we have to change the API key

41

**5.5 Limitations of the solution**

> ●Limited training data: The quality and quantity of training data have
> a significant impact on the accuracy of predictions. The machine
> learning model may not be able to capture all of the elements that
> influence video popularity if the dataset used to train it is too
> small or too unbalanced.
>
> ●YouTube algorithm updates: YouTube modifies its recommendation
> algorithm often, which could render the machine learning model\'s
> forecasts obsolete. The model must be updated frequently to account
> for adjustments made to the platform\'s algorithm.
>
> ●Predictive power of features: The features that were used to train
> the machine learning model may not have the best predictive power when
> it comes to video popularity. The model might not be able to account
> for all the intricate elements that influence video popularity, such
> as recent events, popular culture, or erratic user behavior.
>
> ●Limited scope: The model might only be able to forecast the
> popularity of videos on YouTube, and it might not be transferable to
> other video-sharing websites or other industries.
>
> ●Ethical considerations: Machine learning forecasts of video
> popularity raise ethical questions about potential biases in the data
> or projections and the misuse of the predictions.

42

**CHAPTER-6**

**FINDINGS, CONCLUSION, AND FUTURE WORK**

**6.1 Findings**

The quantity of views, likes, dislikes, and comments, as well as the
video\'s duration, category, title, and thumbnail, are perhaps the most
crucial elements in determining how popular a video will be on YouTube.

The project\'s machine learning models (such as linear regression,
decision trees, and random forests) can successfully predict video
popularity based on the selected features. However, some models
outperform others.

The accuracy of machine learning models can be considerably increased by
feature selection and data preprocessing methods

**6.2 Conclusion**

For content producers, marketers, and other stakeholders, the research
shows the possibility of utilizing machine learning algorithms to
anticipate the popularity of YouTube videos.

This project will directly aid YouTubers by assisting them in analyzing
the type of content that the audience is most interested in watching.

**6.3 Future Work**

The use of more sophisticated machine learning techniques (such neural
networks) and different feature sets could be investigated in future
studies in order to increase the precision of prediction models for
YouTube video popularity.

To better understand how audience involvement (e.g., subscriber count,
view time) and video content (e.g., keywords, captions, audio quality)
affect video popularity, the project might be expanded to incorporate
new features related to both.

The study might possibly be expanded to analyze data from other social
media sites (such as Instagram and TikTok) to investigate how machine
learning can be used to forecast the popularity of content on various
social media sites.

43

**CHAPTER-7**

**REFERENCES**

\[1\] Gupta, V., Diwan, A., Chadha, C., Khanna, A., & Gupta, D. (2022).
Machine Learning enabled models for YouTube Ranking Mechanism and Views
Prediction. arXiv preprint arXiv:2211.11528.

\[2\] Dou, Hongjian, Wayne Xin Zhao, Yuanpei Zhao, Daxiang Dong, Ji-Rong
Wen, and Edward Y. Chang. \"Predicting the popularity of online content
with knowledge-enhanced neural networks.\" In ACM KDD. 2018.

\[3\] Jiayi Xie, Yaochen Zhu, Zhibin Zhang, Jian Peng, Jing Yi, Yaosi
Hu, Hongyi Liu, and Zhenzhong Chen. 2020. A Multimodal Variational
Encoder-Decoder Framework for Micro-video Popularity Prediction. In
Proceedings of The Web Conference 2020 (WWW \'20). Association for
Computing Machinery, New York, NY, USA, 2542--2548.

\[4\] Zhuoran Zhang, Shibiao Xu, Li Guo, and Wenke Lian. 2023.
Multi-modal Variational Auto-Encoder Model for Micro-video Popularity
Prediction. In Proceedings of the 8th International Conference on
Communication and Information Processing (ICCIP \'22). Association for
Computing Machinery, New York, NY, USA, 9--16.
https://doi.org/10.1145/3571662.3571664

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr class="header">
<th>[5]</th>
<th>Can</th>
<th>Social</th>
<th>Features</th>
<th>Help</th>
<th>Learning</th>
<th>to</th>
<th>Rank</th>
<th>YouTube</th>
<th><blockquote>
<p>Videos?</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

\[6\] Li, Y., Eng, K.X., & Zhang, L. (2019). YouTube Videos Prediction:
Will this video be popular?

\[7\] Bielski, Adam & Trzciński, Tomasz. (2018). Pay Attention to
Virality: Understanding Popularity of Social Media Videos with the
Attention Mechanism. 2398-23982. 10.1109/CVPRW.2018.00309.

\[8\] Trzciński, T., Andruszkiewicz, P., Bocheński, T., & Rokita, P.
(2017). Recurrent Neural Networks for Online Video Popularity
Prediction. *ArXiv, abs/1707.06807*.

\[9\] Gothankar, R., Troia, F.D., Stamp, M. (2022). Clickbait Detection
for YouTube Videos. In: Stamp, M., Aaron Visaggio, C., Mercaldo, F., Di
Troia, F. (eds) Artificial Intelligence for

44

+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| Cybe  | Adv   | in    | I     | Secu  | vol   | 54\.  | Spri  | >     |
| rsecu | ances |       | nform | rity, |       |       | nger, | Cham. |
| rity. |       |       | ation |       |       |       |       |       |
+=======+=======+=======+=======+=======+=======+=======+=======+=======+
+-------+-------+-------+-------+-------+-------+-------+-------+-------+

\[10\] A. Vitadhani, K. Ramli and P. Dewi Purnamasari, \"Detection of
Clickbait Thumbnails on YouTube Using Tesseract-OCR, Face Recognition,
and Text Alteration,\" *2021 International Conference on Artificial
Intelligence and Computer Science Technology (ICAICST)*, Yogyakarta,
Indonesia, 2021, pp. 56-61, doi: 10.1109/ICAICST53116.2021.9497811.

\[11\] Agrawal, A. (2016). Clickbait detection using deep learning.
*2016 2nd International Conference on Next Generation Computing
Technologies (NGCT)*, 268-272.

\[12\] S. Zannettou, S. Chatzis, K. Papadamou and M. Sirivianos, \"The
Good, the Bad and the Bait: Detecting and Characterizing Clickbait on
YouTube,\" *2018 IEEE Security and Privacy Workshops (SPW)*, San
Francisco, CA, USA, 2018, pp. 63-69, doi: 10.1109/SPW.2018.00018.

\[13\] B. Gamage, A. Labib, A. Joomun, C. H. Lim and K. Wong,
\"Baitradar: A Multi-Model Clickbait Detection Algorithm Using Deep
Learning,\" *ICASSP 2021 - 2021 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP)*, Toronto, ON, Canada,
2021, pp. 2665-2669, doi: 10.1109/ICASSP39728.2021.9414424.

\[14\] Aditya Khosla, Atish Das Sarma, and Raffay Hamid. 2014. What
makes an image popular? In Proceedings of the 23rd international
conference on World wide web (WWW \'14). Association for Computing
Machinery, New York, NY, USA, 867--876.
https://doi.org/10.1145/2566486.2567996

\[15\] S. Magandran, \"Analysis of YouTube thumbnails: A deep neural
decision forest implementation,\" in Proceedings of the 2019
International Conference on Computing, Mathematics and Engineering
Technologies (iCoMET), 2019, pp. 1-6. DOI: 10.1109/ICOMET.2019.8741923.

\[16\] H. Jang, S. H. Kim, J. S. Jeon, and J. Oh, \"Visual Attributes of
Thumbnails in Predicting Top YouTube Brand Channels: A Machine Learning
Approach,\" in Proceedings of the 2022 IEEE

<table style="width:100%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr class="header">
<th>International</th>
<th>Conference</th>
<th>on</th>
<th>Consumer</th>
<th>Electronics</th>
<th>(ICCE),</th>
<th>2022,</th>
<th>pp.</th>
<th>1-6.</th>
<th><blockquote>
<p>DOI:</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

10.1109/ICCE53269.2022.9717645.

45

\[17\] Park, J. (2022). "The impact of YouTube\'s thumbnail images and
view counts on users\' selection

of video clip, memory recall, and sharing intentions of thumbnail
images". ProQuest Dissertations Publishing. (AT A GLANCE: Journal
Articles, Dissertations, Theses).

\[18\] Y. Chen, Y. Wang, and R. Tan, \"Classifying YouTube Videos by
Thumbnail,\" CS 230 Deep Learning Final Project, Stanford University,
2017. \[Online\].

\[19\] E. Cho, J. Eom and D. Cho, \"Thumbnail Image Design to Grab Views
in Online Video Platform: Evidence from YouTube,\" 2019 International
Conference on KMIS (KMIS), Dublin, Ireland, 2019, pp. 17-22, doi:
10.5220/0008162900170022.

46
