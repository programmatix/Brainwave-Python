import json


def is_custom_object(obj):
    return hasattr(obj, "__dict__")


def snake_to_camel(word):
    components = word.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_custom_object(obj):
            # Convert object attributes from snake_case to camelCase
            obj_dict = self.convert_object_to_dict(obj)
            return {snake_to_camel(k): v for k, v in obj_dict.items()}
        return super().default(obj)

    def convert_object_to_dict(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_object_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_object_to_dict(v) for v in obj]
        elif is_custom_object(obj):
            return {k: self.convert_object_to_dict(v) for k, v in obj.__dict__.items()}
        else:
            return obj
