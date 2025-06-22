This web application generates markdown documents and Python code to implement the specified graph.  Each artifact depends on the previous artifacts.

<table>
  <tr>
    <td style="padding: 8px 15px 8px 0;">
      <b>state-spec.md</b> -- Markdown specification of  the langgraph State Class and its Fields.
    </td>
  </tr>
  <tr>
    <td style="padding: 8px 15px 8px 0;"><b>state_code.py</b> -- Python code implementing the State Class.
    </td>
  </tr>
  <tr>
    <td colspan="2" style="padding: 8px 15px 8px 0;"><b>node-spec.md</b> -- Markdown specification of the langgraph Node Functions.
    </td>
  </tr>
  <tr>
    <td colspan="2" style="padding: 8px 15px 8px 0;"><b>node_code.py</b> -- Python code implementing the langgraph Node Functions.
    </td>
  </tr>
  <tr>
    <td colspan="2" style="padding: 8px 15px 8px 0;"><b>graph_code.py</b> -- Python code implementing the langgraph Graph Builder code and Conditional Edge Functions.
    </td>
  </tr>
  <tr>
    <td colspan="2" style="padding: 8px 15px 8px 0;"><b>main.py</b> -- Python code for running the compiled graph from command line.
    </td>
  </tr>
</table>

To run your graph:

```bash
python {graph_name}/main.py
```

Human input is either through the questionary package or using langgraph interrupt (default).  This is a command line option to main:

```bash
python {graph_name}/main.py --human questionary
```

Using questionary allows graph development without the checkpointer, and without an external program that starts when graph interrupted, gets input, then recreates context for resuming graph.  

```bash
python {graph_name}/main.py --human interrupt
```

Using interrupt and a checkpointer is better adapted to real human use, where response time is unreliable, and ability to resume graph is required for reliability.
