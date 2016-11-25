from itertools import izip
from functools import wraps


def accepts_method(*types):
    """
    top-level decoration, consumes parameters
    """

    def decorator(func):
        """
        actual decorator function, consumes the input function
        """

        @wraps(func)
        def check_accepts(self, *args):
            """
            actual wrapper which does some magic type-checking
            """
            # check if length of args matches length of specified types
            assert len(args) == len(types), "{} arguments were passed to func '{}' but {} " \
                                            "types were passed to decorator '@accepts_method'" \
                .format(len(args), func.__name__, len(types))

            # check types of arguments
            for i, arg, typecheck in izip(range(1, len(args)+1), args, types):
                assert isinstance(arg, typecheck), "type checking: argument #{} was expected to be {} but is {}" \
                    .format(i, typecheck, type(arg))

            return func(self, *args)

        return check_accepts

    return decorator



def accepts_func(*types):
    """
    top-level decoration, consumes parameters
    """

    def decorator(func):
        """
        actual decorator function, consumes the input function
        """

        @wraps(func)
        def check_accepts(*args):
            """
            actual wrapper which does some magic type-checking
            """
            # check if length of args matches length of specified types
            assert len(args) == len(types), "{} arguments were passed to func '{}' only {} " \
                                            "types were passed to decorator '@accepts_func'" \
                .format(len(args), func.__name__, len(types))

            # check types of arguments
            for i, arg, typecheck in izip(range(1, len(args)+1), args, types):
                assert isinstance(arg, typecheck), "type checking: argument #{} was expected to be {} but is {}" \
                    .format(i, typecheck, type(arg))

            return func(*args)

        return check_accepts

    return decorator
