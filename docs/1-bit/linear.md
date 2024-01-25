Before you read this, you need to read:<br>
[operator](operator.md)

## Linear

In normal, we are calulating the linear by using this:

```
y = ax + b
y = (a * x) + b
```

But in 1-bit, we need faster, so we try to make it to:

```
y = (a & x) ^ b
```
