## Operators

Currently we are using 4 of these bitwise operator to work with this one-bit machine learning library:

1. NOT
2. AND
3. XOR
4. OR

## Convert from arithmetic to bitwise.

Because this project is using one bit data type. We can convert from arithmetic operators to bitwise operators. Because bitwise operators are faster than arithmetic operators.

| Arithmetic | Bitwise | Example | Converted |
|:-:|:-:|:-----:|:-----:|
| + | ^ | a + b | a ^ b |
| - | ^ | a - b | a ^ b |
| * | & | a * b | a & b |

## Why you need to convert that operator.

### 1. add and subtract

Look at the binary operators table below, we can see the last digit of number represent of binary digits (char)

| a | b | a + b | a - b | a ^ b |
|:-:|:-:|:-:|:-:|:-:|
|0000000**0**|0000000**0**|0000000**0**|0000000**0**|0000000**0**|
|0000000**0**|0000000**1**|0000000**1**|1111111**1**|0000000**1**|
|0000000**1**|0000000**0**|0000000**1**|0000000**1**|0000000**1**|
|0000000**1**|0000000**1**|0000001**0**|0000000**0**|0000000**0**|

If you do not know what that binary table means, you can see below table (decimal):

| a | b | a + b | a - b | a ^ b |
|:-:|:-:|:-:|:-:|:-:|
|0|0|0|0|0|
|0|1|1|-1|1|
|1|0|1|1|1|
|1|1|2|0|0|

In the binary table, we see the last digit of operator "+", "-", and "^" are the same, so we took the last binary digit to calculate the  the binary.


### 2. multiply (product)

Look at the binary operators table below, we can see the last digit of number represent of binary digits (char)

| a | b | a * b | a & b |
|:-:|:-:|:-:|:-:|
|0000000**0**|0000000**0**|0000000**0**|0000000**0**|
|0000000**0**|0000000**1**|0000000**0**|0000000**0**|
|0000000**1**|0000000**0**|0000000**0**|0000000**0**|
|0000000**1**|0000000**1**|0000000**1**|0000000**1**|

If you do not know what that binary table means, you can see below table (decimal):

| a | b | a * b | a & b |
|:-:|:-:|:-:|:-:|
|0|0|0|0|
|0|1|0|0|
|1|0|0|0|
|1|1|1|1|

In the binary table, we see the last digit of operator "*" and "&" are the same, so we took the last binary digit to calculate the  the binary.
