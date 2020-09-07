""""""
from typing import Dict
import importlib


class StructGenerator:
    """Struct生成器"""

    def __init__(self, filename: str, prefix: str):
        """Constructor"""
        self.filename: str = filename
        self.prefix: str = prefix
        self.typedefs: Dict[str, str] = {}

        self.load_constant()

    def load_constant(self) -> None:
        """"""
        module_name = f"{self.prefix}_typedef"
        module = importlib.import_module(module_name)

        for name in dir(module):
            if "__" not in name:
                self.typedefs[name] = getattr(module, name)

    def run(self):
        """运行生成"""
        self.f_cpp = open(self.filename, "r")
        self.f_struct = open(f"{self.prefix}_struct.py", "w")

        for line in self.f_cpp:
            self.process_line(line)

        self.f_cpp.close()
        self.f_struct.close()

        print("Struct生成成功")

    def process_line(self, line: str) -> None:
        """处理每行"""
        line = line.replace("\n", "")

        if line.startswith("struct"):
            self.process_declare(line)
        elif line.startswith("{"):
            self.process_start(line)
        elif line.startswith("}"):
            self.process_end(line)
        elif "\t" in line and "//" not in line:
            self.process_member(line)

        elif ";" in line and "//" in line:
            self.process_member_add(line)

    def process_declare(self, line: str) -> None:
        """处理声明"""
        words = line.split(" ")
        name = words[1]
        end = "{"

        new_line = f"{name} = {end}\n"
        self.f_struct.write(new_line)

    def process_start(self, line: str) -> None:
        """处理开始"""
        pass

    def process_end(self, line: str) -> None:
        """处理结束"""
        new_line = "}\n\n"
        self.f_struct.write(new_line)

    def process_member(self, line: str) -> None:
        """处理成员"""
        line = line.replace(";", "")
        words = line.split("\t")
        words = [word for word in words if word]

        word = words[0]
        if " " in word:
            py_type = self.typedefs[word.split(" ")[0]]
            name = word.split(" ")[-1]
        else:
            py_type = self.typedefs[word]
            name = words[1]

        new_line = f"    \"{name}\": \"{py_type}\",\n"
        self.f_struct.write(new_line)

    def process_member_add(self, line: str) -> None:
        """处理额外成员"""
        line = line.replace(";", "")
        line = line.replace("\t", "")
        words = line.split(" ")
        words = [word for word in words if word]

        py_type = self.typedefs[words[0]]

        if "//" in words[1]:
            name = words[1].split("//")[0]
        else:
            name = words[1]

        new_line = f"    \"{name}\": \"{py_type}\",\n"
        self.f_struct.write(new_line)


if __name__ == "__main__":
    generator = StructGenerator("../include/sgit/SgitFtdcUserApiStruct.h", "sgit")
    generator.run()
