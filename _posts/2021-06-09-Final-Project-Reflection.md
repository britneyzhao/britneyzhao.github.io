---
layout: post
title: Final Project Reflection
---

Throughout the quarter, I worked alongside my group, Puppy Party, consisting of Kyle Fang, Charisse Hung, and Adhvaith Vijay, to create a [webapp](https://pic16b-dog-detector.herokuapp.com/) for those looking to learn more about dog breeds! Specifically, we created a model that can predict a dog's breed based on a photo, and implemented a dog breed recommendation algorithm, where we recommend three dog breeds based on user-inputted attributes.

To conclude this project, we reflected on our final product with the following reflection questions. We wrote the responses to 1-4 together, and I wrote the answers for 5-6 myself.

## 1. Overall, what did you achieve in your project?

We achieved a variety of goals in these projects. Namely, we created a model capable of predicting over 120 dog breeds based on an image. Based on 10 epochs of training we achieved validation accuracy in the range of 80%. On top of our dog breed classifier we also created an interactive web page for users to find out what dog breeds most align with their interest. Using categories such as ease of maintenance, dog size, and trainability we are able to gauge which dog breed(s) are best suited for an individual.

## 2. What are two aspects of your project that you are especially proud of?

One feature of our project is that the model learns from incorrect breed predictions when the user submits a Google Form containing the picture and correct dog breed. This is one portion of our project that we are particularly proud of, as the model actively learns from data beyond the original dataset we trained our model on. When a user inputs the incorrectly identified photo alongside the correct dog breed, this data is added to a spreadsheet that our model can then train with and learn from. On the web application, this learning process automatically occurs once a day, so within 24 hours, our model will implement the correct prediction of the same photo. We thought having the model learn from its mistakes was an innovative addition to our project and are especially proud of it for this reason. 

We are also very proud of the dog breed recommendation portion of our project. On the “Find Your Perfect Dog!” page of our web application, users can input a variety of attributes they want in a dog, and our project then returns the top three matches based on a dataset we found online. We made this page as user-friendly as possible, with an efficient matching algorithm to ensure a fast result, as well as hyperlinks that users can click on and go to the American Kennel Club website containing more information about the matched breeds. Users can then easily learn more about these dog breeds and make an informed decision regarding which breed best suits their lifestyle. 

## 3. What are two things you would suggest doing to further improve your project? (You are not responsible for doing those things.)

One suggestion to further improve our project is to include more dog breeds. Currently, the breed prediction model is trained on 121 breeds and the breed recommender includes 199 breeds. There are many more dog breeds that could be included in these two aspects of our project. Furthermore, our breed prediction model is trained on images of purebred dogs, and thus it does not perform as well on mixed breed dogs. If we could obtain or create a database of mixed breed dogs that included the list of breeds that each dog is, we could further train the model to predict the multiple breeds of a dog. This does pose many challenges since the amount of breed combinations is very great, however, it would allow our project to be more inclusive of dogs.

Another suggestion to improve our project would be to add a ranking system to the breed recommender. Currently, the user selects their preferences on 6 dog features, and all of these features are weighted equally when recommending the dog breeds. It is likely that of the 6 presented features, a user may care about some features more than others. Perhaps they want a dog with minimal shedding and maintenance, but don’t care about the size of the dog. Adding a ranking system for the features would provide recommendations that better match the preferences of the user. 

## 4. How does what you achieved compare to what you set out to do in your proposal? (if you didn't complete everything in your proposal, that's fine!)

We completed more than what we included in our proposal! Our project proposal only indicated a model on a webapp that would input a picture and output the dog breed. We successfully did this and added more features. 
Our model also includes an online learning feature. The model can take feedback from the user and improve itself in the next run. We also included sample images, so people who don’t have photos handy on their devices can still enjoy the webapp. We also included a dog recommender tab where the user can input his or her preferences, and the webapp will use KD Trees to predict the top 3 matches and display corresponding pictures and links.

## 5. What are three things you learned from the experience of completing your project? Data analysis techniques? Python packages? Git + GitHub? Etc?

This class was actually my first time using Git and GitHub, so through this project, I became very familiar with the group project functionalities through pushing and pulling commits from the origin. Using GitHub, our group was able to work on the project on our own time while contributing to the code at our own pace. We could also see when our group members updated the GitHub repository, allowing us to clearly see how we were progressing through our project. 

I also learned more about the Python package SciPy when creating our dog breed recommendation algorithm. Before this class, I had zero experience with this Python library, and our project enabled me to learn more about the functionality of this package. I initially struggled with figuring out how to recommend a dog breed based on various attributes, but after I read more about SciPy, I found that SciPy has functions that can do exactly what I wanted with KDTree. Thus, overall, this project helped me understand how resourceful Python libraries can be. 

I also learned a lot about how machine learning works overall. In my PIC16A class, our class never touched on any type of machine learning, so I was very excited to learn it in PIC16B. While I am not an expert in the field, the introduction to TensorFlow in class gave me a good foundation to expand my understanding of machine learning more through our group project. Seeing the photo identification model come to life and accurately identify dog breeds was a very rewarding experience, and I really enjoyed learning about how to create these types of models with our project. 

## 6. How will your experience completing this project will help you in your future studies or career? Please be as specific as possible.
This group project allowed me to learn how to effectively communicate and work with multiple people while online. It can be difficult to consistently update other group members when we cannot see each other in person regularly. Since this was one of my first experiences working with a group solely online, this group project will greatly help me in the future as online communication becomes more and more common with those separated physically due to distance. 

In addition to effective communication, this group project showed me various ways to collaborate on a coding project with multiple people. I originally was not familiar with GitHub and Google Colab, so learning how these applications work will make future collaborative coding projects easier. Learning and using GitHub effectively will be a very useful skill in the future, as publicly showing projects and code can be a useful resource for others. 

For skills specifically related to coding in Python, I learned how to use very important packages like `pandas`, `numpy`, and `scipy` that make data analysis much easier. While I was familiar with some of these packages from PIC16A, I now feel very comfortable using these Python libraries since we consistently used them throughout our project. 

