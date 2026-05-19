"""Retroactive E2E tests for Phase 0 rendering and event-handling fixes.

Covers:

* formatCell - integers render as integers (with thousands separator),
  floats with two decimals, null/undefined as empty string. Phase 0
  fixed the blanket toFixed(2) that turned "Times Displayed: 1234" into
  "1234.00".

* renderTable - never uses innerHTML interpolation of user data. Phase 0
  rewrote the function to use DOM APIs so a CSV cell like
  '<img src=x onerror=alert(1)>' does not execute when rendered.

* showTab - takes an explicit event parameter instead of the implicit
  global ``event``, and uses ``event.currentTarget`` so the active
  class always lands on the button element.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def _bring_page_up(page, analysis_url: str) -> None:
    page.goto(analysis_url)
    page.wait_for_function("typeof formatCell === 'function' && typeof renderTable === 'function'")


# ----------------------------------------------------------------------
# formatCell
# ----------------------------------------------------------------------


class TestFormatCell:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (1234, "1,234"),
            (1000000, "1,000,000"),
            (-42, "-42"),
        ],
    )
    def test_integers_render_without_decimals(self, page, analysis_url, value, expected) -> None:
        _bring_page_up(page, analysis_url)
        result = page.evaluate("v => formatCell(v)", value)
        assert result == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.5, "1.50"),
            (3.14159, "3.14"),
            (-2.75, "-2.75"),
            (0.001, "0.00"),
        ],
    )
    def test_floats_render_with_two_decimals(self, page, analysis_url, value, expected) -> None:
        _bring_page_up(page, analysis_url)
        result = page.evaluate("v => formatCell(v)", value)
        assert result == expected

    @pytest.mark.parametrize(
        "value",
        [None, ""],
    )
    def test_null_and_empty_render_as_empty_string(self, page, analysis_url, value) -> None:
        _bring_page_up(page, analysis_url)
        result = page.evaluate("v => formatCell(v)", value)
        assert result == ""

    def test_strings_pass_through(self, page, analysis_url) -> None:
        _bring_page_up(page, analysis_url)
        result = page.evaluate("v => formatCell(v)", "Item Alpha")
        assert result == "Item Alpha"

    def test_infinity_and_nan_render_as_strings(self, page, analysis_url) -> None:
        _bring_page_up(page, analysis_url)
        # We pass values via evaluate; Infinity / NaN are serialized,
        # so we use a small wrapper that returns them inline.
        inf_result = page.evaluate("() => formatCell(Infinity)")
        nan_result = page.evaluate("() => formatCell(NaN)")
        assert inf_result == "Infinity"
        assert nan_result == "NaN"


# ----------------------------------------------------------------------
# renderTable / XSS safety
# ----------------------------------------------------------------------


class TestRenderTableXssSafe:
    def test_cell_containing_script_tag_is_not_executed(self, page, analysis_url) -> None:
        """Inject a row whose cell would, if rendered via innerHTML,
        attempt to execute a script.  Phase 0 replaced innerHTML
        interpolation with textContent, so the script must NOT run."""
        _bring_page_up(page, analysis_url)

        # Mark a global so we can detect if the injected handler fires.
        page.evaluate("window.__xssDetected = false;")

        page.evaluate(
            """
            () => {
              // Create a container so renderTable has somewhere to write.
              const c = document.createElement('div');
              c.id = '__xssProbe';
              document.body.appendChild(c);
              const malicious = [{
                Item: "<img src=x onerror='window.__xssDetected=true'>"
              }];
              renderTable('__xssProbe', malicious);
            }
            """
        )

        # If the renderer used innerHTML, the onerror handler would
        # have run by now. Give it a tick to be safe.
        page.wait_for_timeout(200)
        detected = page.evaluate("() => window.__xssDetected")
        assert detected is False

    def test_cell_containing_script_tag_renders_as_literal_text(self, page, analysis_url) -> None:
        """The malicious markup should appear in the DOM as text content,
        not as parsed HTML."""
        _bring_page_up(page, analysis_url)

        page.evaluate(
            """
            () => {
              const c = document.createElement('div');
              c.id = '__xssProbeText';
              document.body.appendChild(c);
              renderTable('__xssProbeText', [{ Item: "<b>not bold</b>" }]);
            }
            """
        )
        # textContent should contain the literal angle brackets.
        text = page.text_content("#__xssProbeText td")
        assert text == "<b>not bold</b>"
        # There should be no <b> element inside the cell.
        bold_count = page.locator("#__xssProbeText td b").count()
        assert bold_count == 0


# ----------------------------------------------------------------------
# showTab
# ----------------------------------------------------------------------


class TestShowTab:
    def test_clicking_a_tab_marks_it_active(self, page, analysis_url) -> None:
        _bring_page_up(page, analysis_url)
        # Reveal the results container so the tabs are interactable.
        page.evaluate("() => document.getElementById('resultsContainer').classList.add('show')")
        # Click the "Display Balance" tab.
        page.evaluate(
            "() => Array.from(document.querySelectorAll('.tab-btn'))"
            ".find(b => b.textContent.trim() === 'Display Balance').click()"
        )
        active_label = page.evaluate(
            "() => document.querySelector('.tab-btn.active').textContent.trim()"
        )
        assert active_label == "Display Balance"

    def test_only_one_tab_button_active_at_a_time(self, page, analysis_url) -> None:
        _bring_page_up(page, analysis_url)
        page.evaluate("() => document.getElementById('resultsContainer').classList.add('show')")
        # Click two tabs in sequence; only the last should remain active.
        page.evaluate(
            "() => Array.from(document.querySelectorAll('.tab-btn'))"
            ".find(b => b.textContent.trim() === 'Selection Frequencies').click()"
        )
        page.evaluate(
            "() => Array.from(document.querySelectorAll('.tab-btn'))"
            ".find(b => b.textContent.trim() === 'Data Tables').click()"
        )
        active_count = page.evaluate("() => document.querySelectorAll('.tab-btn.active').length")
        assert active_count == 1
        active_label = page.evaluate(
            "() => document.querySelector('.tab-btn.active').textContent.trim()"
        )
        assert active_label == "Data Tables"

    def test_showtab_does_not_throw_without_implicit_event_global(self, page, analysis_url) -> None:
        """The pre-fix code relied on the deprecated implicit window.event.
        Confirm showTab still works when invoked with an event argument."""
        _bring_page_up(page, analysis_url)
        # Call directly with a synthetic event.
        result = page.evaluate(
            """
            () => {
              try {
                const btn = document.querySelector('.tab-btn');
                showTab({ currentTarget: btn }, 'scores');
                return 'ok';
              } catch (e) {
                return 'threw: ' + e.message;
              }
            }
            """
        )
        assert result == "ok"
