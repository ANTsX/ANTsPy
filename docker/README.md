# ANTsPy docker

## ANTsPy build

The latest commit of ANTsPy is built from source. The build arg `antspy_version`
can be used to build a specific release or commit.


## Controlling threads

The default number of threads is 1, override by passing the
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` variable, eg

```
docker run -e ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 ...
```


## Version and dependency information

Run the container with `-m pip freeze` to get version information.
