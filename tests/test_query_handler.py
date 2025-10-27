'''Test query handler functionality'''

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_help.query_handler import QueryHandler
from cli_help.config import Config

class TestQueryHandler(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.query_handler = QueryHandler(self.config)

    def test_process_simple_query(self):
        '''Test processing a simple query'''
        response = self.query_handler.process_query("how do I list files?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_list_tools(self):
        '''Test listing available tools'''
        tools = self.query_handler.list_available_tools()
        self.assertIsInstance(tools, list)

if __name__ == '__main__':
    unittest.main()
