# Motor-Imagery-BCI
A Motor Imagery based Brain Computer Interface capable of classifying four classes of Motor Imagery (the mental simulation of physical movement) using Common Spatial Patterns based features and XGBoost classifiers.

#### Brain Computer Interface (BCI)
Brain Computer Interfaces are devices that enable humans to interact and communicate with devices by understanding and modelling brain activity. 

#### Motor Imagery
Motor Imagery is the mental simulation or imagination of physical movement. This causes an Event Related Synchronisation or Event Related De-Synchronisation of EEG signals in various frequency bands. These patterns can be used to identify and distnguish between various classes of motor imagery (e.g. movement of left versus right hand)

A common class of BCIs are those that use Motor Imagery to control external devices. For example, recognizing that a person is imagining right hand movement can be used as a signal to control a prosthetic arm.

 This project develops a BCI capable of classifying four classes of Motor Imagery using Common Spatial Patterns based features. 
 
 #### Common Spatial Patterns (CSP)

CSP is used to transform the filtered EEG signals belonging to two different classes such that the variance in the signal representing the first class is maximised, while the variance in the signal representing the second class is minimised. Here we use the <b>log variance of the transformed signals</b> as features.

#### Dataset 
We use the <a href='http://www.bbci.de/competition/iv/'> BCI Competition 2008 – Graz data set A </a> to train our models.

#### Multiclass CSP

Since the method of Common Spatial Patterns is only applicable to binary classification problems, we use three classification paradigms to extend CSP to a mutli-class classification problem -

<ul>
  <li> 
    <b>One versus the Rest</b> - Four classifiers are trained, one for each class in the dataset. The final decision is that of the classifier that makes a prediction with the highest probability
  </li>
  <li>
    <b>One versus One</b> - A binary classifier is trained for each pair of classes in the dataset. The classifier corresponding to "Left versus Right hand" and "Tongue versus Foot" are first invoked. The decision of these two classifiers decides which binary classifier makes the final decision. e.g. if the "Left Hand versus Right Hand" classifier predicts Left Hand movement and the "Tongue versus Foot" classifier predicts Tongue movement, the "Left Hand versus Tongue" classifier decides the class of the trial. 
  </li>
  <li>
    <b>Grouped</b> - This classification paradigm employs two levels of classification. The first level distinguishes between trials that correspond either to Left or Right Hand movement OR to Tongue or Foot Movement. If the trial is classified as belonging to one of Left or Right Hand movement, the Left versus Right Hand classifier is invoked to classify the trial, else the Tongue versus Foot movement is invoked to classify the trial
  </li>
</ul>
