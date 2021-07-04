#RTlib# 
##introduction##
European Emission Authority (EEA) have been carrying out CO2 emission test for heavy duty vehicles. This is mainly done by using specific test suit known as VECTO test and each vehicle run on set of test assigned to it’s category according to the Regulation (EU)  2018/956. 
In this paper, a novel machine learning based approach been evaluate to predict “Specific CO2 emission” by considering manufacture specifications and without running VECTO test.


##build instruction##

* **to run test suite**
    * excecute:  <tb> *python setup.py pytest*<br>
    this will run tests folder codes
* **to build library**
    * run: *python setup.py bdist_wheel*
* **to install** 
    * install to local machine: *pip install ./dist/<###.whl>*
