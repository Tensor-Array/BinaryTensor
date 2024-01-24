## Calculating loss

Normally, you are calculating loss by:

```
loss = pow((a - b), 2)
loss = (a - b) * (a - b)
```

We can change it to bitwise version if you are using 1 bit datatype.

```
loss = (a ^ b) & (a ^ b)
```

If (`a` is `0` or `1`) and (`b` is `0` or `1`)
