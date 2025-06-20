This web application generates markdown documentation and langgraph code for the selected graph specification.

<table>
  <tr>
    <td style="padding: 8px 15px 8px 0; width: 250px;"><button style="background-color: #333333; color: white; border: none; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 8px; width: 100%;">state spec</button></td>
    <td style="vertical-align: middle;">
      Documents the fields needed by the graph's state (`{graph_name}/state-spec.md`)
    </td>
  </tr>
  <tr><td colspan="2" style="padding-bottom: 5px;">{state_spec_status}</td></tr>
  <tr>
    <td style="padding: 8px 15px 8px 0;"><button style="background-color: #333333; color: white; border: none; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 8px; width: 100%;">state code</button></td>
    <td style="vertical-align: middle;">
      Translates the state specification into a Python class. (`{graph_name}/state_code.py`)
    </td>
  </tr>
  <tr><td colspan="2" style="padding-bottom: 5px;">{state_code_status}</td></tr>
  <tr>
    <td style="padding: 8px 15px 8px 0;"><button style="background-color: #333333; color: white; border: none; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 8px; width: 100%;">node spec</button></td>
    <td style="vertical-align: middle;">
      Creates the Markdown specification for the node functions. (`{graph_name}/node-spec.md`)
    </td>
  </tr>
  <tr><td colspan="2" style="padding-bottom: 5px;">{node_spec_status}</td></tr>
  <tr>
    <td style="padding: 8px 15px 8px 0;"><button style="background-color: #333333; color: white; border: none; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 8px; width: 100%;">node code</button></td>
    <td style="vertical-align: middle;">
      Translates the node specification into Python functions. (`{graph_name}/node_code.py`)
    </td>
  </tr>
  <tr><td colspan="2" style="padding-bottom: 5px;">{node_code_status}</td></tr>
  <tr>
    <td style="padding: 8px 15px 8px 0;"><button style="background-color: #333333; color: white; border: none; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 8px; width: 100%;">graph code</button></td>
    <td style="vertical-align: middle;">
      Creates langgraph graph builder code and conditional edge functions. (`{graph_name}/graph_code.py`)
    </td>
  </tr>
  <tr><td colspan="2" style="padding-bottom: 5px;">{graph_code_status}</td></tr>
  <tr>
    <td style="padding: 8px 15px 8px 0;"><button style="background-color: #333333; color: white; border: none; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; cursor: pointer; border-radius: 8px; width: 100%;">main</button></td>
    <td style="vertical-align: middle;">
      Creates python program that runs graph from the command line. (`{graph_name}/main.py`)
    </td>
  </tr>
  <tr><td colspan="2" style="padding-bottom: 5px;">{main_status}</td></tr>
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