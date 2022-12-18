Outline
=====================

Guideline
~~~~~~~~~~~~~~~~~~~
Here is what you should take care for your implementations:

- **Be careful to change parameters in other classes.**
  Because of python system, all parameters can be changeable.
  However, it doesn't mean that you should change them.
  Only referring those parameters outside the class is somehow reasonable, but changing them is dangerous for some situations.
  The names of parameters that should not be changed outside are started from "_".
  Therefore, the parameter whose name is started from "_" is highly forbidden to access outside the class.
- **Publicly accessible parameters and methods don't mean you should access them.**
  For some convenience, we make some parameters and methods are public starting without "_".
  However, if you should access them is always depending on what you want to do.
  Therefore, some accessible parameters and methods could not be appropriate to be used correspondence to the actual markets.

TBD...
