from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyOverview:
    title: str
    body_html: str
    links: tuple[tuple[str, str], ...]


FAMILY_OVERVIEWS: dict[str, FamilyOverview] = {
    "Alkali Metal": FamilyOverview(
        title="Alkali metals",
        body_html=(
            "<p>Alkali metals have a single electron in their outer s subshell. "
            "That lone valence electron is easily lost, so these elements form +1 ions "
            "and react vigorously with water and oxygen.</p>"
            "<p>They are soft, low-density metals with low melting points that increase down the group.</p>"
        ),
        links=(
            ("LibreTexts", "https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_(Zumdahl_and_Decoste)/12%3A_Chemistry_of_the_Nonmetals/12.1%3A_The_Alkali_Metals"),
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/group/1"),
        ),
    ),
    "Alkaline Earth Metal": FamilyOverview(
        title="Alkaline earth metals",
        body_html=(
            "<p>Alkaline earth metals have two valence electrons in the outer s subshell. "
            "They commonly form +2 ions and produce basic oxides and hydroxides.</p>"
            "<p>They are harder and less reactive than alkali metals, with higher melting points.</p>"
        ),
        links=(
            ("LibreTexts", "https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_(Zumdahl_and_Decoste)/12%3A_Chemistry_of_the_Nonmetals/12.2%3A_The_Alkaline_Earth_Metals"),
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/group/2"),
        ),
    ),
    "Transition Metal": FamilyOverview(
        title="Transition metals",
        body_html=(
            "<p>Transition metals fill d subshells and often have multiple stable oxidation states. "
            "Partially filled d orbitals lead to colored compounds, magnetic behavior, and catalytic activity.</p>"
            "<p>They form coordination complexes and are central to many industrial and biological processes.</p>"
        ),
        links=(
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/typical/transition-metal"),
            ("Britannica", "https://www.britannica.com/science/transition-element"),
        ),
    ),
    "Post-Transition Metal": FamilyOverview(
        title="Post-transition metals",
        body_html=(
            "<p>Post-transition metals are softer, lower-melting metals to the right of the d block. "
            "They tend to form covalent or mixed ionic-covalent bonds and have relatively low electrical conductivity.</p>"
            "<p>Common examples include aluminum, tin, and lead.</p>"
        ),
        links=(
            ("LibreTexts", "https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_(Zumdahl_and_Decoste)/12%3A_Chemistry_of_the_Nonmetals/12.3%3A_The_Post-Transition_Metals"),
            ("Wikipedia", "https://en.wikipedia.org/wiki/Post-transition_metal"),
        ),
    ),
    "Metalloid": FamilyOverview(
        title="Metalloids",
        body_html=(
            "<p>Metalloids have properties between metals and nonmetals. "
            "Their bonding and conductivity can be tuned by temperature or doping.</p>"
            "<p>Many metalloids are semiconductors used in electronics and photovoltaics.</p>"
        ),
        links=(
            ("Britannica", "https://www.britannica.com/science/metalloid"),
            ("LibreTexts", "https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_(Zumdahl_and_Decoste)/12%3A_Chemistry_of_the_Nonmetals/12.4%3A_The_Metalloids"),
        ),
    ),
    "Nonmetal": FamilyOverview(
        title="Nonmetals",
        body_html=(
            "<p>Nonmetals tend to gain or share electrons to complete their valence shells. "
            "They form molecular compounds with covalent bonding and show a wide range of states of matter.</p>"
            "<p>Nonmetals include essential life elements such as carbon, nitrogen, and oxygen.</p>"
        ),
        links=(
            ("LibreTexts", "https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_(Zumdahl_and_Decoste)/12%3A_Chemistry_of_the_Nonmetals/12.5%3A_The_Nonmetals"),
            ("Britannica", "https://www.britannica.com/science/nonmetal"),
        ),
    ),
    "Halogen": FamilyOverview(
        title="Halogens",
        body_html=(
            "<p>Halogens are highly reactive nonmetals with seven valence electrons. "
            "They tend to gain one electron to form -1 ions and are strong oxidizing agents.</p>"
            "<p>They exist as diatomic molecules (for example, F2 and Cl2) under standard conditions.</p>"
        ),
        links=(
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/group/17"),
            ("LibreTexts", "https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_(Zumdahl_and_Decoste)/12%3A_Chemistry_of_the_Nonmetals/12.7%3A_The_Halogens"),
        ),
    ),
    "Noble Gas": FamilyOverview(
        title="Noble gases",
        body_html=(
            "<p>Noble gases have complete valence shells, which makes them very unreactive. "
            "They are monoatomic gases with low boiling points and are used in lighting and inert atmospheres.</p>"
            "<p>Heavier noble gases can form a limited number of compounds under specific conditions.</p>"
        ),
        links=(
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/group/18"),
            ("Britannica", "https://www.britannica.com/science/noble-gas"),
        ),
    ),
    "Lanthanide": FamilyOverview(
        title="Lanthanides",
        body_html=(
            "<p>Lanthanides fill 4f subshells and are often called rare-earth elements. "
            "They commonly form +3 ions and show strong magnetic and optical behavior.</p>"
            "<p>They are used in magnets, lasers, and phosphors.</p>"
        ),
        links=(
            ("Britannica", "https://www.britannica.com/science/lanthanide"),
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/term/lanthanides"),
        ),
    ),
    "Actinide": FamilyOverview(
        title="Actinides",
        body_html=(
            "<p>Actinides fill 5f subshells and are all radioactive. "
            "Many exhibit multiple oxidation states and complex chemistry.</p>"
            "<p>Several actinides (such as uranium and plutonium) are important in nuclear science.</p>"
        ),
        links=(
            ("Britannica", "https://www.britannica.com/science/actinide"),
            ("Royal Society of Chemistry", "https://www.rsc.org/periodic-table/term/actinides"),
        ),
    ),
}


def family_overview_html(family: str) -> str | None:
    if not family:
        return None
    entry = FAMILY_OVERVIEWS.get(family)
    if not entry:
        return None
    links = " ".join(
        f"<a href=\"{href}\">{label}</a>" for label, href in entry.links
    )
    return (
        "<div style=\"line-height:1.4;\">"
        f"<p><b>{entry.title} overview</b></p>"
        f"{entry.body_html}"
        f"<p><b>Learn more:</b> {links}</p>"
        "</div>"
    )
