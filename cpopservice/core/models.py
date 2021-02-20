import bson


class BaseModel(object):
    def __init__(self, params={}):
        self.__dict__ = params

    def to_bson(self):
        result = bson.dumps(self.__dict__)
        return result

    @classmethod
    def from_bson(cls, value):
        result = bson.loads(value)
        obj = cls()
        obj.__dict__.update(result)
        return obj

    def __eq__(self, other):
        return other and self.__dict__ == other.__dict__

    def __hash__(self, other):
        # TODO: implement hash function if needed
        return 0

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.__dict__)


class Event(BaseModel):
    # attributes: timestamp, shape (list of Coordinates), position (Coordinates)
    pass


class Coordinates(BaseModel):
    pass