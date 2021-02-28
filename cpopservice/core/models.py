import abc
import json

import bson


class Serializable(abc.ABC):

    def to_bson(self) -> bytes:
        raise NotImplementedError


class BaseModel(Serializable):
    def __init__(self, params={}):
        self.__dict__ = params

    def to_bson(self) -> bytes:
        # result = bson.dumps(self.__dict__)
        result = json.dumps(self.__dict__).encode('UTF-8')
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
