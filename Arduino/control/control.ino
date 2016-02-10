/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.
 
  This example code is in the public domain.
 */
 
// Pin 13 has an LED connected on most Arduino boards.
// give it a name:
int C1B = 12;
int C1A = 8;
int C2B = 4;
int C2A = 2;

char data;
// the setup routine runs once when you press reset:
void setup() {                
  // initialize the digital pin as an output.
  Serial.begin(9600);
  pinMode(C1A, OUTPUT);     
  pinMode(C1B, OUTPUT);
  pinMode(C2A, OUTPUT);
  pinMode(C2B, OUTPUT);
}

void turn_left()
{
  digitalWrite(C1A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C1B, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2B, HIGH);   // turn the LED on (HIGH is the voltage level)
}

void turn_right()
{
  digitalWrite(C1A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C1B, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2A, HIGH);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2B, LOW);   // turn the LED on (HIGH is the voltage level)
}

void fwd()
{
  digitalWrite(C1A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C1B, HIGH);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2B, HIGH);   // turn the LED on (HIGH is the voltage level)
}

void back()
{
  digitalWrite(C1A, HIGH);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C1B, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2A, HIGH);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2B, LOW);   // turn the LED on (HIGH is the voltage level)
}

void Stop()
{
  digitalWrite(C1A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C1B, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2A, LOW);   // turn the LED on (HIGH is the voltage level)
  digitalWrite(C2B, LOW);   // turn the LED on (HIGH is the voltage level)
}

void loop() {
  if(Serial.available()>0)
  {
    data = Serial.read();
    
    Serial.println(data);
    
    switch(data)
    {
      case 'l':
              turn_left();
              delay(100);               // wait for a second
              Stop();
              break;
      case 'r':
              turn_right();
              delay(100);               // wait for a second
              Stop();
              break;
      
      case 'f':
              fwd();
              delay(100);               // wait for a second
              Stop();
              break;
      case 'b':
              back();
              delay(100);               // wait for a second
              Stop();
              break;
      
    }
  }
}
