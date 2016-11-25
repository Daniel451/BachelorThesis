# Bachelor Thesis - Daniel Speck

This repository contains the bachelor thesis of Daniel Speck (PDF)
as well as code examples that can be used for any academic purpose.
Please remember to cite the work.

The thesis proposes a new approach for localizing the ball in
RoboCup humanoid soccer. A deep neural architecture is used without
any preprocessing at all. The localization part gets solved by a
convolutional neural network that is trained with probability
distributions on full images. No sliding-window approach is used.
The thesis only proposes a concept for the tracking part, consisting of
a simple recurrent neural network that filters and stabilizes the
localization output and predicts future distributions.


### Balltracking for Robocup Soccer using Deep Neural Networks - Bachelor Thesis

The complete thesis can be viewed or downloaded here:
[thesis_ball_tracking_speck.pdf](https://gogs.mafiasi.de/12speck/BachelorThesis/src/master/thesis_ball_tracking_speck.pdf)




### Ball Localization for Robocup Soccer using Convolutional Neural Networks - Paper

The complete paper can be viewed or downloaded here:
[paper_ball_localization_speck.pdf](https://gogs.mafiasi.de/12speck/BachelorThesis/src/master/paper_ball_localization_speck.pdf)


##### Cite - Bibtex

```
@inproceedings{speck2016robocup,
  title={Ball Localization for Robocup Soccer using
Convolutional Neural Networks},
  author={Speck, Daniel and Barros, Pablo and Weber, Cornelius and Wermter, Stefan},
  booktitle={RoboCup 2016: Robot World Cup XX},
  year={2016},
  series={Lecture Notes in Computer Science},
  organization={Springer International Publishing}
}
```
