# TubeTrend-Pro-Predicting-YouTube-Video-Popularity


## What are we doing in the project??

The objective of this project is to develop a machine learning model
capable of predicting the expected number of views for a forthcoming
YouTube video that a content creator plans to publish in the near
future. This will be a significant challenge as it requires analyzing
multiple factors that contribute to a video\'s popularity, such as
title, thumbnail and past performances of other videos of the same
channel.


## Brief Description of the Solution Approach
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


## Why we are doing what we are doing??
1.The thumbnail is the first thing that viewers see when they come across
a video on YouTube, and it can have a significant impact on whether they
click on the video or not. A visually appealing and attention-grabbing
thumbnail can increase the likelihood of a viewer clicking on the video
and watching it. Creators should aim to create thumbnails that
accurately represent the content of the video while also being visually
appealing and attention-grabbing.

2.The title of a YouTube video is also crucial in determining its
popularity. A clear and descriptive title that accurately reflects the
content of the video can make it easier for viewers to find and watch
the video. Additionally, creators can use clickbait titles to generate
curiosity and intrigue among viewers, which can increase the likelihood
of the video being clicked on and watched. However, it\'s important to
ensure that the title accurately reflects the content of the video to
avoid disappointing viewers and damaging the creator\'s reputation.

Since thumbnail and title of a video plays major role in getting views hence extra attention has been given to measure this factor.

3.Also the channels ability to garner views also depends upon the amount of reach the channel has . Which could be measured by subscribers count.
But we observed that this may not be a very good parameter because some youtube channels get more views than their subscriber count while this may not be true for other youtube channels.
Hence we took a youtube channels performances on its last videos as a measure of reach the channel has.
And to do so we took performance of last 10 videos.


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
