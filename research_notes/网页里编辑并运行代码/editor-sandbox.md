# Editor widget and execution sandbox

Research date: 1 November 2026. Older product posts are marked as older. This note is for one private learner on their own Mac, not a public contest judge. It stays at the level of editor, request, worker, time limit, and captured output. It does not describe ways to break a sandbox.

## Which editor libraries are current, what a minimal embed looks like, and what the license is?

### Takeaway

As of November 2026 the two libraries built for dropping a code box into a page are Monaco Editor (release 0.57.0, September 2026, MIT) and CodeMirror 6 (MIT). Ace is still published (1.45.2, October 2026, BSD-3-Clause). None of them run Python. They only edit text in the page. Python highlighting in Monaco 0.57.0 is a language definition; the package’s language-service workers are for other languages.

### Cited Findings

- Monaco Editor is the browser code editor from VS Code. The npm package `monaco-editor` is version 0.57.0, license MIT, and the registry listing was updated on 24 September 2026. — [npm monaco-editor](https://www.npmjs.com/package/monaco-editor)
- The GitHub changelog entry for that release is “Prepare Monaco Editor 0.57.0 release,” dated 24 September 2026. The default branch still had commits after that release, including 30 October 2026. — [microsoft/monaco-editor](https://github.com/microsoft/monaco-editor/)
- The license text is the MIT License, copyright 2016–present Microsoft Corporation. — [LICENSE.txt](https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt)
- Install is `npm install monaco-editor`. The package ships an ESM build under `/esm` and a versioned `monaco.d.ts`. The README says the AMD build is deprecated and will be removed. — [README.md](https://github.com/microsoft/monaco-editor/blob/main/README.md)
- The documented minimal create, from the webpack-plugin guide, is an ESM import plus one call. The sample uses JavaScript; the same `language` option accepts a bundled language id:

```javascript
import * as monaco from 'monaco-editor';
// or import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';

monaco.editor.create(document.getElementById('container'), {
  value: 'console.log("Hello, world")',
  language: 'javascript'
});
```

  — [monaco-editor webpack-plugin README](https://github.com/microsoft/monaco-editor/blob/main/webpack-plugin/README.md)
- Monaco Editor 0.57.0’s `esm/metadata.js` lists a language whose label is `python`, with entry `vs/languages/definitions/python/register` and no `worker` field. TypeScript, JSON, HTML, and CSS entries in the same file do include a worker. — [monaco-editor@0.57.0 metadata](https://cdn.jsdelivr.net/npm/monaco-editor@0.57.0/esm/metadata.js)
- The `codemirror` npm package is an example configuration over the `@codemirror` packages. The listing checked here is 6.0.2, license MIT. Its documented setup is:

```javascript
import {EditorView, basicSetup} from "codemirror"

const view = new EditorView({
  parent: document.body,
  doc: "Hello",
  extensions: [basicSetup]
})
```

  The same guide says `view.state.doc.toString()` returns the current document. — [codemirror on npm](https://www.npmjs.com/package/codemirror); [Basic Editor example](https://codemirror.net/examples/basic/)
- `basicSetup` bundles line numbers, history, bracket matching, autocompletion, search, and the default highlight style. `minimalSetup` is a smaller set: default keymap, history, special-character highlighting, custom selection drawing, and the default highlight style. The basic example says a language package is added as another extension. — [codemirror package docs via jsDocs](https://www.jsdocs.io/package/codemirror); [Basic Editor example](https://codemirror.net/examples/basic/)
- Python support is a separate package, `@codemirror/lang-python`. The registry listing checked here is 6.2.1, license MIT. The changelog dates 6.2.1 to 14 May 2025. The npm page’s “updated” timestamp is 13 April 2026 while the version number shown is still 6.2.1. — [@codemirror/lang-python on npm](https://www.npmjs.com/package/@codemirror/lang-python); [changelog](https://github.com/codemirror/lang-python/blob/main/CHANGELOG.md); [LICENSE](https://github.com/codemirror/lang-python/blob/main/LICENSE)
- The JavaScript language example on the CodeMirror site is `javascript({typescript: true})` added next to `basicSetup`. The Python package is the matching extension for Python; the basic-example page does not paste a Python snippet. — [Basic Editor example](https://codemirror.net/examples/basic/); [@codemirror/lang-python](https://www.npmjs.com/package/@codemirror/lang-python)
- Ace (Ajax.org Cloud9 Editor) is still published. `ace-builds` and `ace-code` are version 1.45.2, license BSD-3-Clause, registry listings updated 21 October 2026. — [ace-builds](https://www.npmjs.com/package/ace-builds); [ace-code](https://www.npmjs.com/package/ace-code)
- Ace’s documented embed is a sized, positioned element plus one script call. The README says the element must be explicitly sized and `position: absolute` or `relative`:

```html
<div id="editor">some text</div>
<script src="src/ace.js" type="text/javascript" charset="utf-8"></script>
<script>
    var editor = ace.edit("editor");
</script>
```

  Languages other than plain text are separate modules. The README’s concrete example is `src/mode-javascript.js`. — [ajaxorg/ace README](https://github.com/ajaxorg/ace?tab=readme-ov-file)
- A maintainer reply on 2 July 2019 says Ace is distributed under the BSD 3-clause license and can be used commercially. The 1.4.1 license text is the BSD license with the Ajax.org name-endorsement clause. — [ajaxorg/ace issue 4001](https://github.com/ajaxorg/ace/issues/4001); [LICENSE at v1.4.1](https://github.com/ajaxorg/ace/blob/v1.4.1/LICENSE)
- Thebe, which turns static HTML into runnable cells, tells authors to put `data-language="python"` on a `<pre data-executable="true">` “to enable syntax highlighting with CodeMirror.” That is an editor inside a larger runner, not a fourth general-purpose editor library. — [Thebe minimal example](https://thebe.readthedocs.io/en/latest/examples/minimal_example.html)

### Inferences

- For a single lesson page, CodeMirror 6 is the smaller embed: one `EditorView`, extensions added only as needed, and a documented string read-back. Monaco is the closer match to the VS Code editing surface, at the cost of its loader and worker setup. Ace remains a maintained script-tag embed if the page should avoid a module bundler.
- Monaco’s bundled Python mode is for editing and coloring. Execution, tracebacks, and pass/fail are outside the editor package.
- A 2026 npm “updated” date is not the same thing as a new release. `@codemirror/lang-python` still shows 6.2.1, whose changelog date is May 2025.

### Gaps

- The Monaco typedoc page for the method that reads the buffer back (`getValue` is the usual name) was not fetched. The sourced create sample shows the initial `value` option only.
- Ace’s README example names the JavaScript mode file. This pass did not fetch the Python mode filename.
- No newer `@codemirror/lang-python` release than 6.2.1 was found. A release published after the 13 April 2026 npm page timestamp would not be covered if the registry page was stale.
- Thebe’s own license page was not fetched.

## How do LeetCode, CodeSignal, or similar products isolate a Run click?

### Takeaway

The shared shape, where a vendor actually describes it, is: the editor stays in the browser; Run sends source to a service; a worker runs that source outside the web server; a time limit stops a long run; stdout, stderr, and a verdict come back to the same page. LeetCode’s own help page describes the page and the judging result, and does not describe the worker. CodeSignal’s first-party posts are older and describe two different mechanisms: a Monaco editor talking to a language server, and a long-lived Docker session for an interview. Replit’s April 2026 post is the current first-party account of per-user isolation, and it is a multi-tenant product.

### Cited Findings

- LeetCode’s help article “Start your Coding Practice,” shown as updated 17 November 2022, says the question page has a code editor, and “Many thanks to the CodeMirror Team for helping us make this possible.” It says Run Code executes against the current test case and the console then shows input, “your output,” and expected output. Compile errors and other errors appear in the Run Code Result panel. Submit sends the code to “our online judging system,” which evaluates it against a large set of backend test cases “to ensure it meets various time and space restrictions.” Results land in the Submissions tab. — [LeetCode Help Center, 17 November 2022](https://support.leetcode.com/hc/en-us/articles/360012016874-Start-your-Coding-Practice)
- That 2022 article does not name the queue, the worker, the sandbox program, or the timeout implementation. A later system-design writeup says outright that LeetCode publishes almost nothing about its own judge, and therefore discusses isolate, Judge0, gVisor, and Firecracker as published systems a judge can be built from. — [Designing LeetCode, KnowSys](https://knowsys.tech/topics/74-design-leetcode)
- LeetCode’s own release notes, a Discuss post updated through the 2026.5 section and carrying dates from 4 September 2024 to 5 June 2026, describe editor preferences, accessibility, focus mode, and contest replay. They do not name Monaco or CodeMirror, and they do not describe the judge process. — [Feature Release Notes](https://leetcode.com/discuss/post/5736503/feature-release-notes-by-leetcode-9awg/)
- Third-party browser extensions describe the live LeetCode editor as Monaco and mention CodeMirror as a legacy or alternate mode. These are not LeetCode statements. One extension README says LeetCode uses Monaco and also uses CodeMirror in some editor modes. A font-changer note says it targets “Monaco (the LeetCode editor) and CodeMirror (legacy).” A 2024 Vim extension says LeetCode uses Monaco. — [LeetSense](https://www.harxit.com/products/leetsense); [leetcode-font-changer](https://github.com/shivam23rawat/leetcode-font-changer); [vim-leetcode](https://github.com/samlevine03/vim-leetcode)
- CodeSignal’s IDE post, reprinted on Medium with the date 5 March 2021, says the frontend is based on Monaco and `monaco-languageclient`, that a language server starts as a separate process, and that a WebSocket carries editor traffic. For live interviews it says each user gets a Docker container so language-server processes stay separate. This post is about completion and navigation, not about the Run worker. — [CodeSignal autocompletion post](https://codesignal.com/blog/how-we-support-autocompletion/); [Medium tag page dating the essay 5 March 2021](https://medium.com/codesignal/tagged/programming)
- An older CodeSignal post, “The power of persistence,” says the earlier design was an IDE plus “distributed services for running code,” and that they later run a whole interview in one Docker container left up until the session ends, with client traffic over a WebSocket. The fetched article text does not print a publication date. Its cover image URL on the page is under `/2022/12/`, and the body discusses the COVID-19 shift to remote interviews, so this is an older description. It says persistent sessions were, at the time of writing, used for free-coding tasks. — [CodeSignal persistent sessions](https://codesignal.com/blog/the-power-of-persistence-building-a-new-architecture-for-interviews/)
- CodeSignal’s current GraphQL docs for a filesystem task include `timeLimit` in milliseconds and `unitTests` “for scorable environments.” The support article on advanced questions says the environment is a preconfigured Docker container, and that each test-taker session is hosted in a Docker container. It also states session time limits (2 hours in Interview unless refreshed, 8 hours in Assessments). Those limits are session limits, not the per-run `timeLimit` field. — [FilesystemExecutionEnvironmentInput](https://developer.codesignal.com/graphql/types/filesystemexecutionenvironmentinput.doc.html); [Create an advanced assessment question](https://support.codesignal.com/hc/en-us/articles/7385597441943-Create-an-advanced-assessment-question)
- Replit’s 20 April 2026 post, updated 21 April 2026, says each user’s development environment runs in its own isolated sandbox, built on Linux containers hardened with seccomp-bpf and other layers. It says Linux containers are not a perfect isolation boundary because they share a kernel, and that Replit is rolling out microVMs so the guest does not share the host kernel. Some users already have the microVM path. — [Defense in Depth, Replit, 20 April 2026](https://replit.com/blog/defense-in-depth-how-replit-secures-every-layer-of-the-vibe-coding-stack)
- A Replit post from 18 March 2024 describes the browser connection as a WebSocket proxied to the machine that runs Conman, “our container manager that runs all Repls in a container.” — [More Reliable Connections to Your Repls, 18 March 2024](https://replit.com/blog/eval)
- Judge0, an open-source system in this same product category, documents the split explicitly: a submission is created and “waits in queue to be processed”; the web API and the worker can be different hosts; the worker is what processes submissions. The default compose file runs `server`, `worker`, Postgres, and Redis, with `privileged: true` on the server and worker. Configuration comments say Judge0 uses isolate as the sandbox. — [Judge0 CE API docs](https://ce.judge0.com/docs); [judge0.conf](https://github.com/judge0/judge0/blob/master/judge0.conf); [docker-compose.yml](https://raw.githubusercontent.com/judge0/judge0/master/docker-compose.yml)
- Judge0 returns `stdout`, `stderr`, `compile_output`, and a `status`. Documented status ids include 1 In Queue, 2 Processing, 3 Accepted, 4 Wrong Answer, 5 Time Limit Exceeded, 6 Compilation Error, 11 Runtime Error (NZEC), 12 Runtime Error (Other). `cpu_time_limit` is in seconds and counts assigned CPU time. `wall_time_limit` is wall-clock seconds; the docs say it should be set much higher than the CPU limit “as a precaution against sleeping programs.” If `expected_output` is null, stdout is not compared. — [Judge0 CE API docs](https://ce.judge0.com/docs)
- Piston’s v2 execute API takes `language`, `version`, and `files`, plus optional `stdin`, `run_timeout`, `compile_timeout`, and memory limits. The response includes `run.stdout`, `run.stderr`, `run.output`, `run.code`, and `run.signal`. The readthedocs page’s prose for `run_timeout` and `compile_timeout` appears to describe the other stage; the field names themselves are the reliable part of that page. — [Piston API v2](https://piston.readthedocs.io/en/latest/api-v2/)

### Inferences

- A Run click on these products is not “the editor executes the buffer.” The editor is a text control. Something behind it starts a process with a time bound and returns captured streams plus a verdict.
- For one trusted student, the part worth copying is that split and the time bound, so a tight loop cannot freeze the page. The multi-tenant machinery (a queue, a privileged isolate worker, a container per session, a microVM) exists because those sites run strangers’ code on a shared host.
- LeetCode’s 2022 help text and the 2024 extension claims disagree about the editor library. Treat CodeMirror as what LeetCode’s own help still said in 2022, and Monaco as what unofficial clients observed later. There is no 2024–2026 LeetCode statement in the sources above that settles it.
- CodeSignal’s 2021 Docker-per-user note is about language servers for the editor. Its persistent-session post is about where the interview’s files and terminal live. Those are different boxes from a short Run-and-grade call.
- Pass/fail is a verdict computed from output or tests. Judge0 can do the stdout comparison itself (`expected_output` → Accepted or Wrong Answer). Piston returns streams and an exit code and does not, in the cited API, return Accepted or Wrong Answer.

### Gaps

- No first-party LeetCode engineering post from 2024–2026 was found that names the sandbox, the queue, or the per-run timeout.
- The CodeSignal persistence article has no visible publication date in the fetched text. It should not be cited as a 2024–2026 design.
- Replit’s April 2026 post does not describe the timeout or the stdout path of a single Run click. The March 2024 post describes the connection proxy, not grading.
- Third-party “how LeetCode works” posts that specify SQS, gVisor, or Firecracker as LeetCode’s stack were not used as facts about LeetCode. KnowSys presents them as design choices, and says LeetCode has not published the judge.

## What local or self-hosted options exist for one student on their own Mac?

### Takeaway

Four real options fit a private Python lesson. A localhost service that runs the project’s own virtualenv is the small one and can import the same packages as the rest of the course. Pyodide, including JupyterLite, runs in the browser and needs no execution server, but it is a separate Python. Judge0 and Piston are self-hosted HTTP execution services aimed at untrusted code; both want Docker, and both are ongoing services to patch. A plain local process is not a contest sandbox. It is enough when the only person who can click Run is the person who owns the Mac.

### Cited Findings

- `subprocess.run(..., capture_output=True, timeout=...)` captures the child process’s stdout and stderr. If `timeout` seconds elapse, the child is killed and `TimeoutExpired` is raised after the child has terminated. The docs also say process creation itself may not be interruptible, so the exception is not guaranteed until after creation finishes. With `check=True`, a non-zero exit raises `CalledProcessError`, whose attributes include the exit code and the captured streams. `text=True` makes the captured streams strings. — [Python subprocess docs](https://docs.python.org/3/library/subprocess.html)
- The Python tutorial shows an unhandled exception printed as a stack traceback whose last line is the exception type and message (`ZeroDivisionError`, `NameError`, `TypeError` in the examples). It says the preceding lines are the context, “in the form of a stack traceback,” and that this display “will not display lines read from standard input.” — [Python tutorial, Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html)
- Pyodide’s stable quickstart loads `https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.js`, calls `loadPyodide()`, and runs a string with `pyodide.runPython(...)`. It says that after import, only the standard library is available until more packages are loaded. The same page’s complete example is one HTML file with a text field, a Run button, and an output area; the Run handler `try`/`catch`es and writes the error into that area. — [Pyodide quickstart](https://pyodide.org/en/stable/usage/quickstart.html)
- Pyodide’s `setStdout` / `setStderr` redirect standard streams. In the browser the default stdout handler is described as a batched callback that logs. A batched handler receives either a full line or a flushed partial line. — [Redirecting standard streams](https://pyodide.org/en/stable/usage/streams.html)
- The Pyodide license file on the main branch is the Mozilla Public License 2.0. A larger work may be distributed under other terms as long as the Pyodide source-form obligations are kept. — [pyodide LICENSE](https://raw.githubusercontent.com/pyodide/pyodide/main/LICENSE)
- The same quickstart shows Python reading the page through `import js` and creating DOM nodes. That is the opposite of a separate execution host: page and interpreter share one browser. — [Pyodide quickstart](https://pyodide.org/en/stable/usage/quickstart.html)
- JupyterLite “runs entirely in the browser” and “requires no dedicated application server much less a container orchestrator.” It can be served as static files, locally or on a static host, and it is embeddable. Python kernels listed there run in a Web Worker via Pyodide or Xeus Python. Notebook files are saved in the browser’s IndexedDB or localStorage. The homepage compatibility table marks jupyterlite-core 0.7.0 and 0.6.0 as supported and says only the last two releases are actively supported. The same page’s changelog navigation lists 0.8.x and 0.9.0 pre-releases, so that table and the changelog index do not describe the same tip. — [JupyterLite](https://jupyterlite.readthedocs.io/en/latest/)
- Thebe keeps code in the lesson HTML (`<pre data-executable="true" data-language="python">`) and runs it on a Jupyter kernel. The minimal example requests a kernel from mybinder.org and warns that the first Run waits while Binder starts. The configuration reference also shows a local server, with `kernelOptions.serverSettings.baseUrl` illustrated as `http://127.0.0.1:8888` and a token. — [Thebe minimal example](https://thebe.readthedocs.io/en/latest/examples/minimal_example.html); [Thebe configuration reference](https://thebe.readthedocs.io/en/stable/config_reference.html); [Thebe “get started”](https://thebe.readthedocs.io/en/latest/start.html)
- Judge0 is GPL-3.0. The project says it is self-hostable, speaks a JSON HTTP API, sandboxes compilation and execution, and supports on the order of 90 languages in the current README (an older docs page says 60+). The README’s cloud example posts `language_id`, `source_code`, and `stdin` to `/submissions?wait=true`. The docs say `wait=true` “may or may not be enabled,” and that on the official host it is not enabled; a self-hosted config has to be checked. — [judge0/judge0](https://github.com/judge0/judge0/); [LICENSE](https://github.com/judge0/judge0/blob/master/LICENSE); [Judge0 CE API docs](https://ce.judge0.com/docs)
- The Judge0 compose file pulls `judge0/judge0:latest` twice (server on port 2358, and a worker), plus `postgres:16.2` and `redis:7.2.4`. Server and worker set `privileged: true`. — [docker-compose.yml](https://raw.githubusercontent.com/judge0/judge0/master/docker-compose.yml)
- Judge0’s compilers image says its sandbox is Isolate, “licensed under GPL v2,” and that Isolate uses Linux kernel features, “like namespaces and control groups.” — [judge0/compilers](https://github.com/judge0/compilers)
- An isolate discussion, with comments from 2017 and a 7 May 2023 update, says running isolate in Docker needs the privileged flag because it creates cgroups, and the 2023 note still says the privileged flag is required. Those comments also discuss host cgroup settings. That is a Linux-host operational constraint, not a Mac guide. — [ioi/isolate issue 35](https://github.com/ioi/isolate/issues/35)
- Piston’s GitHub repository page describes a code execution engine and lists the license as MIT. A fetch of the raw `LICENSE` and `README.md` URLs returned 404 in this pass, so the license claim rests on the repository page metadata returned with the project description. The same page content says the public API is no longer freely available as of 15 February 2026, that keys are not granted for individual projects, and that people can run their own instance. — [engineer-man/piston](https://github.com/engineer-man/piston/)
- Piston’s documented host requirements are Docker, Docker Compose, Node.js 15 or newer for the CLI, cgroup v2 enabled, and cgroup v1 disabled. `docker-compose up -d api` starts an API with no language runtimes until the CLI installs them. The API listens on port 2000. Package downloads use `PISTON_REPO_URL`, defaulting to a GitHub release index, so installing runtimes wants a network once. — [Piston repository page](https://github.com/engineer-man/piston/); [Piston configuration](https://piston.readthedocs.io/en/latest/configuration/); [Deployment notes via DeepWiki](https://deepwiki.com/engineer-man/piston/5-deployment)
- Piston does not document a pass/fail field. The execute response is language, version, and the run (and optional compile) stream objects. A lesson would compare `run.stdout` or `run.code` itself. — [Piston API v2](https://piston.readthedocs.io/en/latest/api-v2/)

### Inferences

- For this course’s Mac, a localhost runner pointed at the existing `.venv` is the option that can execute the same Python the notebooks already use. Pyodide and JupyterLite do not launch that virtualenv. The quickstart says a fresh Pyodide has the standard library only, and the interpreter is the one loaded in the browser.
- That localhost runner is a network service only in the narrow sense: the browser calls `127.0.0.1` while the page is open. It does not need a public host, an account, or a queue. Maintenance is the small script plus the virtualenv the course already has. `timeout` covers a stuck child process; it does not create a separate filesystem or a separate user.
- A single Docker container around that same virtualenv adds a maintenance cost (Docker Desktop, image updates) and a stronger boundary than `subprocess`. This pass did not find a vendor guide aimed at “one student, one container,” so that option is an inference from ordinary Docker use, not a cited product.
- Judge0’s maintenance cost is high for one learner: four containers, a database, Redis, privileged mode, and a Linux sandbox that the compilers page ties to kernel namespaces and cgroups. GPL-3.0 matters if the lesson app is distributed with Judge0 modifications. It is a poor fit when the only code being run is the student’s own.
- Piston is lighter than Judge0 (no Postgres in the cited install) and MIT, but it is still a Docker service with cgroup requirements written for a Linux host. No Mac-specific install page was found. Its public API is not an available backend for a personal project after 15 February 2026, according to the repository text.
- Thebe-on-Binder is a network service run by mybinder.org, with a cold start. Thebe-on-localhost is a local Jupyter server, which is the same family as “a local HTTP service,” with the kernel protocol instead of a one-shot `/execute`.
- Pass/fail is a few lines beside any of these runners: compare captured stdout with the expected text, or run a checker and read its exit code. Judge0 is the option in this list that already returns Accepted and Wrong Answer.

### Gaps

- No official Judge0 or Piston document in this pass says the compose stack runs on Docker Desktop for Mac. Linux cgroup and privileged-container notes should not be read as a tested Mac recipe.
- The Piston raw license file was not retrieved (404). The MIT label is from the GitHub repository metadata in the search result.
- Whether Pyodide 0.314 can load the scientific packages a later lesson needs was not checked. The quickstart only says extra packages must be loaded and that the standard library is the starting set.
- JupyterLite’s license was not fetched. The homepage table and the changelog index disagree about how far past 0.7 the supported line has moved.
- `subprocess` killing the direct child is documented. This pass did not fetch a page on what happens to grandchildren the child may have spawned. That matters for a hostile program and is a weak concern for the student’s own exercises.
- The tutorial shows the traceback text. This pass did not fetch `sys.excepthook`, so the notes do not claim a separate sentence that names `sys.stderr` as the stream. Capturing the child stderr is still the documented way to collect the child process’s error stream.

## How do these products keep the statement, the editor, and the result on one page?

### Takeaway

The editor is a widget inside the lesson document, not a second application. Run writes into another region of that same document. The learner does not switch tools because nothing navigates away. LeetCode’s 2022 help text, Thebe’s HTML cells, and Pyodide’s one-file sample are three documented versions of that layout.

### Cited Findings

- LeetCode’s 2022 help article describes a single question page: the statement sections, the editor, and Run Code. After execution, “the console will automatically expand” on that page and shows input, the learner’s output, the expected output, and compile or other errors. Submit results are shown in the Submissions tab, still part of the question flow the article is describing. — [LeetCode Help Center, 17 November 2022](https://support.leetcode.com/hc/en-us/articles/360012016874-Start-your-Coding-Practice)
- Monaco and CodeMirror are created inside a DOM node the page already owns (`document.getElementById('container')`, `parent: document.body`). The surrounding HTML can be the problem statement. The result panel is ordinary page markup updated when the run returns. — [Monaco webpack-plugin README](https://github.com/microsoft/monaco-editor/blob/main/webpack-plugin/README.md); [CodeMirror basic example](https://codemirror.net/examples/basic/)
- Thebe’s model is a static lesson page. The author writes the explanation as HTML and marks code blocks with `data-executable="true"`. Activating Thebe turns those blocks into live cells; Run renders output on that page. The project’s own description is “turns your static HTML pages into interactive ones, powered by a kernel.” — [Thebe](https://thebe.readthedocs.io/en/latest/); [minimal example](https://thebe.readthedocs.io/en/latest/examples/minimal_example.html)
- Pyodide’s quickstart alternative is one `index.html`: instructions, an input, a Run button, and a textarea labeled Output. — [Pyodide quickstart](https://pyodide.org/en/stable/usage/quickstart.html)
- JupyterLite can embed a live REPL in a website, and it can also be the whole JupyterLab or Notebook UI. The full UI is a different tool surface from a paragraph-plus-editor lesson, even though it is still one origin. — [JupyterLite](https://jupyterlite.readthedocs.io/en/latest/)
- CodeSignal’s advanced-question help describes the test-taker working in the question’s environment: filesystem, terminal, and IDE limits are properties of that session. It does not spell out a three-pane layout. The 2021 IDE post says the editing UI is Monaco in their cloud IDE, which is one interview surface rather than a handoff to a desktop editor. — [Advanced assessment question](https://support.codesignal.com/hc/en-us/articles/7385597441943-Create-an-advanced-assessment-question); [Autocompletion post, 2021](https://codesignal.com/blog/how-we-support-autocompletion/)
- Ace’s embed is the same pattern: a `div` in the page becomes the editor. — [Ace README](https://github.com/ajaxorg/ace?tab=readme-ov-file)

### Inferences

- “One page” is a front-end choice. The statement, the widget, and the result element are siblings in one HTML document. The runner can be Pyodide in the page, a localhost process, or a remote worker; the layout does not change.
- A Jupyter notebook already is one document of prose, code, and output. Thebe and an embedded Pyodide box are the versions of that idea that look like a reading page with a Run button. JupyterLite’s full Lab UI is closer to switching into an IDE, which the question wants to avoid.
- The pass/fail line belongs in the result region next to stdout and the traceback. LeetCode’s 2022 console shows output beside the expected output. A private lesson can show the same three things without a submissions history.

### Gaps

- No 2024–2026 LeetCode screenshot or help revision was found that reconfirms the console layout. The layout claim rests on the 17 November 2022 article.
- CodeSignal’s posts describe one IDE session. They were not fetched in a form that itemizes “statement pane, editor pane, result pane.”
- This pass did not measure bundle size or startup time of Monaco versus CodeMirror on a 16 GB Mac.
