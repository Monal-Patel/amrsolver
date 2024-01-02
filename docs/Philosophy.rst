
**********************
Philosophy
**********************
Development philosophy


=====================
1. Code
=====================

* https://lefticus.gitbooks.io/cpp-best-practices/content/
* Minimise memory use. Currently, 4*cons fields (S1,S2,Sborder,dudt) + prims fields stored currently (+ temporary fabarrays for flux computations, however, these are insignificant compared to whole field). Low storage arbitary stage order SSPRK (optimal) time integration.
* Clang-Format with google style

=====================
2. Testing
=====================



=====================
3. Documentation
=====================

-----------------
ReStructuredText
-----------------

* https://bashtage.github.io/sphinx-material/rst-cheatsheet/rst-cheatsheet.html

.. Title
.. =====
.. Titles are underlined (or over- and underlined) with
.. a nonalphanumeric character at least as long as the
.. text.

.. A lone top-level section is lifted up to be the
.. document's title.

.. Any non-alphanumeric character can be used, but
.. Python convention is:

.. * ``#`` with overline, for parts
.. * ``*`` with overline, for chapters
.. * ``=``, for sections
.. * ``-``, for subsections
.. * ``^``, for subsubsections
.. * ``"``, for paragraphs