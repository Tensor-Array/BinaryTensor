Before you read this, you need to read:<br>
[operator](operator.md)

## Calculating loss

Normally, you are calculating loss by:

```
loss = pow((a - b), 2)
loss = (a - b) * (a - b)
```

We can change it to bitwise operator if you are using 1 bit datatype.

```
loss = (a ^ b) & (a ^ b)
```

#### If `a` and `b` are only one bit:

```
if (a = b):
    loss = 0
else:
    loss = 1
```
