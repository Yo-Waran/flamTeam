import 'package:flutter/material.dart';

//main function
void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: "My Test App", home: HomeWidget());
  }
}

class HomeWidget extends StatefulWidget {
  @override
  State<HomeWidget> createState() => _HomeWidgetState();
}

class _HomeWidgetState extends State<HomeWidget> {
  String testText = "";

  void changeMyText(String text) {
    //this function will be called when the button is pressed
    //this will set the text of the controller
    this.setState(() {
      this.testText = text;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("My First Flutter App"),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          TextInputWidget(this.changeMyText),
          MyButton(),
          Text(
            "Submitted Name : " + this.testText,
            style: TextStyle(fontSize: 25, color: Colors.amberAccent),
          ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: [
          BottomNavigationBarItem(icon: Icon(Icons.pageview), label: "Search"),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings_input_component),
            label: "Settings",
          ),
        ],
        fixedColor: Colors.white,
        backgroundColor: Colors.black,
        unselectedItemColor: Colors.white,
      ),
    );
  }
}

class TextInputWidget extends StatefulWidget {
  //make a callback function
  final Function(String)
  callback; //declare a function that takes a string as an argument

  TextInputWidget(
    //this is the constructor for the TextInputWidget class
    this.callback,
  ); //this will be used to call the function in the parent widget

  @override
  _TextInputWidgetState createState() => _TextInputWidgetState();
}

class _TextInputWidgetState extends State<TextInputWidget> {
  //create a controller
  final controlller = TextEditingController();

  void changeText(String text) {
    //this function set the text of the controller
    if (text == "Hello World") {
      this.controlller.clear(); //this will clear the text field
      text = "";
    }
    setState(() {
      //this will set the state of the widget
      this.controlller.text = text;
    });
  }

  void buttonPress() {
    //this function will be called when the button is pressed
    //'widget' is used to access the parent widget
    widget.callback(
      this.controlller.text,
    ); //this will call the callback function
    this.controlller.clear(); //this will clear the text field
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: <Widget>[
        TextField(
          controller: this.controlller,
          decoration: InputDecoration(
            prefixIcon: Icon(Icons.nature_outlined),
            labelText: "Submit Your Name",
            hintText: "Enter your name here",
            suffixIcon: IconButton(
              icon: Icon(Icons.send),
              onPressed: this.buttonPress,
              splashColor: Colors.red,
              tooltip: "Submits the name",
            ),
          ),
          onChanged: (text) => this.changeText(text),
        ),
        Text(
          "${this.controlller.text}",
        ), //this will show the text of the controller
      ],
    );
  }
}

class MyButton extends StatefulWidget {
  @override
  State<MyButton> createState() => _MyButtonState();
}

class _MyButtonState extends State<MyButton> {
  int counter = 0;
  void buttonPress() {
    //this function will be called when the button is pressed
    setState(() {
      //this will set the state of the widget when the button is pressed
      this.counter = this.counter + 1; //this will increment the counter
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        FloatingActionButton(
          foregroundColor: Colors.blueGrey,
          backgroundColor: Colors.blue,
          onPressed: buttonPress, // this will call the buttonPress function
          child: Icon(Icons.auto_fix_normal_outlined),
        ),
        Text(
          "Counter: ${this.counter}",
          style: TextStyle(fontSize: 20),
        ), //this will show the counter
      ],
    );
  }
}
