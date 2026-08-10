**====================KEYPRESS=CORRELATION====================**

**1)** Record ($\#_1$) of keypresses of each key on my keyboard

**2)** Record myself typing a known phrase \[SAMPLE\]

**3)** Detect transients in \[SAMPLE\]

**4)** **For each transient:**     
&nbsp;&nbsp;&nbsp;&nbsp;**4a)** Run correlation filters on each recording of each possible key  
&nbsp;&nbsp;&nbsp;&nbsp;**4b)** Exclude any outliers from correlation results  
&nbsp;&nbsp;&nbsp;&nbsp;**4c)** Store the ($\#_2$) most correlated keys with correlation values  

**5)** Reconstruct the message by taking the most correlated key for each transient

**6)** _OPTIONAL:_ If the message is invalid (e.g. message or words within message  
are not in a list of known/expected words), run error correction:    
&nbsp;&nbsp;&nbsp;&nbsp;**6a)** Replace a key with relatively low correlation on its most correlated key and  
relatively high correlation on its second-most correlated key with that second-most  
key value.   
&nbsp;&nbsp;&nbsp;&nbsp;**6b)** If the message is still invalid, replace the next most fitting key with its  
second-most correlated key-value. Repeat for ($\#_3$) iterations or until the message  
is valid.  
&nbsp;&nbsp;&nbsp;&nbsp;**6c)** If the message is still invalid after ($\#_3$) iterations, mark the key from  
6a as excluded from this process, reset the message to its original estimated values,  
and repeat steps 6a and 6b for ($\#_4$) iterations or until the message is valid.



**============================================================**

&nbsp;&nbsp;&nbsp;&nbsp;Keyboard samples were recorded in a car using the internal microphone of a   
&nbsp;&nbsp;&nbsp;&nbsp;Zoom H4n handheld recorder. The typing of the secret message was done during    
&nbsp;&nbsp;&nbsp;&nbsp;the same recording session, on the built-in keyboard on a ASUS Vivobook. 