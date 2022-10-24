def _is_object(key):
    return key.split(".")[-1][0].isupper()


class WorkflowSession:
    def __init__(self):
        pass

    def construct(self, object):
        if issubclass(type(object), WorkflowObject):
            if hasattr(object, "WORKFLOW_ID"):  # object already constructed
                return object
            if _is_object(object.WORKFLOW_BC_KEY):
                object.WORKFLOW_ID = generate_random_uuid()
                print(f"CONSTRUCT {object.WORKFLOW_ID} => {parameterize_workflow(object)}")
                return object
            else:
                return self.get_attribute(
                    object.WORKFLOW_PARENT, object.WORKFLOW_BC_KEY.split(".")[-1]
                )
        elif (
            object is None
            or issubclass(type(object), int)
            or issubclass(type(object), float)
            or issubclass(type(object), str)
        ):
            return object
        elif issubclass(type(object), list):
            return [self.construct(x) for x in object]

    def run_method(self, object, method_name, args, kwargs):
        object = self.construct(object)
        args = [self.construct(arg) for arg in args]
        kwargs = {k: self.construct(v) for k, v in kwargs.items()}
        print(
            f"RUN {object.WORKFLOW_ID}.{method_name} with args = {args}, and kwargs = {kwargs}"
        )

    def get_attribute(self, object, attribute_name):
        object = self.construct(object)
        object.attribute_name.WORKFLOW_ID = generate_random_uuid()
        print(
            f"GET_ATTR {object.WORKFLOW_ID}.{attribute_name} => {object.attribute_name.WORKFLOW_ID}"
        )
        return object.attribute_name.WORKFLOW_ID

    def __getattribute__(self, key):
        EXCLUDED_WORDS = ["run_method", "construct", "get_attribute"]
        if key in EXCLUDED_WORDS:
            return object.__getattribute__(self, key)
        return WorkflowObject(key, None, self)


def generate_random_uuid():
    import uuid

    return str(uuid.uuid4())


def parameterize_workflow(object):
    if issubclass(type(object), WorkflowObject):
        if not hasattr(object, "args") or not hasattr(object, "kwargs"):
            print("Warning: WorkflowObject has no args or kwargs")
            print(f"Object is {object.WORKFLOW_BC_KEY}")
        return {
            **(
                {"BC_workflow_id": object.WORKFLOW_ID}
                if hasattr(object, "WORKFLOW_ID")
                else {}
            ),
            **{"BC_class_name": object.WORKFLOW_BC_KEY.split(".")[-1]},
            **{"args": [parameterize_workflow(arg) for arg in object.args]},
            **{"kwargs": {k: parameterize_workflow(v) for k, v in object.kwargs.items()}},
        }
    elif (
        object is None
        or issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
    ):
        return object
    elif issubclass(type(object), list):
        return [parameterize_workflow(x) for x in object]
    else:
        raise ValueError(f"Cannot parameterize {object} of type {type(object)}")


class WorkflowObject:
    def __init__(self, key, parent) -> None:
        self.WORKFLOW_BC_KEY = key
        self.WORKFLOW_PARENT = parent
        self.WORKFLOW_CHILDREN = {}

    def __getattribute__(self, key):
        EXCLUDED_WORDS = [
            "WORKFLOW_BC_KEY",
            "WORKFLOW_SESSION",
            "WORKFLOW_PARENT",
            "WORKFLOW_ID",
            "WORKFLOW_CHILDREN",
            "args",
            "kwargs",
            "shabalaba",
        ]

        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key in self.WORKFLOW_CHILDREN:
                return self.WORKFLOW_CHILDREN[key]
            else:
                self.WORKFLOW_CHILDREN[key] = WorkflowObject(
                    self.WORKFLOW_BC_KEY + "." + key, self, self.WORKFLOW_SESSION
                )
                return self.WORKFLOW_CHILDREN[key]

    def __repr__(self) -> str:
        if hasattr(self, "args") or hasattr(self, "kwargs"):
            return parameterize_workflow(self)
        return "WorkflowObject(" + self.WORKFLOW_BC_KEY + ")"

    def __call__(self, *args, **kwargs):
        if self.WORKFLOW_BC_KEY.split(".")[-1][0].isupper():  # instantiating an object
            self.args = args
            self.kwargs = kwargs
            return self
        else:
            return self.WORKFLOW_SESSION.run_method(
                self.WORKFLOW_PARENT, self.WORKFLOW_BC_KEY.split(".")[-1], args, kwargs
            )